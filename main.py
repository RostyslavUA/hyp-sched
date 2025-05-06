import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from LinSATNet import linsat_layer
from networks import HGNNModel, HGNNModel_2layer
from model import CustomLossBatch, utility_fn
from utils import HyperDataset, get_data, hypergraph_generation, custom_collate_fn # , exh_solver
from conflict_vs_hypergraph import generate_channel_matrix, build_hyperedges, build_conflict_edges


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def data_generate(train_size):
    H_train, hedge_train = [], []
    for i in range(train_size):
        # Generate a random channel matrix
        H = generate_channel_matrix()

        # Build conflict constraints:
        # 1. Pairwise conflict edges (conflict graph)
        conflict_edges = build_conflict_edges(H)
        conflict_edges_list = [list(edge) for edge in conflict_edges]

        # 2. Higher order hyperedges (hypergraph conflicts)
        hyperedges = build_hyperedges(H, conflict_edges)
        hyperedges_list = [list(edge) for edge in hyperedges]
        H_train.append(H)
        hedge_train.append(hyperedges_list)
        if i % 100 == 0:
            print(i)
    return H_train, hedge_train


if __name__ == '__main__':
    # -------------------------------
    # Simulation parameters
    # -------------------------------
    N = 13                   # number of users/links
    area_size = 100.0        # square area side length (meters)
    path_loss_exp = 3.0      # path loss exponent
    P = 50.0                 # transmit power in Watts
    noise_power = 1e-8       # noise power in Watts
    SINR_threshold = 6       # SINR threshold (linear scale, e.g., 10 ~ 10 dB)
    train_size = 10000
    val_size = 100
    generate_data = False

    # Noise vector for all links
    noise_vec = np.full(N, noise_power)
    if generate_data:
        print("Train data generation")
        H_train, hedge_train = data_generate(train_size)
        torch.save({'H': H_train, 'hedges': hedge_train}, 'data/train_data.pt')
        print("Val data generation")
        H_val, hedge_val = data_generate(val_size)
        torch.save({'H': H_val, 'hedges': hedge_val}, 'data/val_data.pt')
        print("Datagen done")
    else:
        data = torch.load('data/train_data.pt', weights_only=False)
        H_train = data['H']
        hedge_train = data['hedges']
        data = torch.load('data/val_data.pt', weights_only=False)
        H_val = data['H']
        hedge_val = data['hedges']
    i = 0
    Is, Dv_invs, De_invs, Hs, Ws = [], [], [], [], []
    for I, hyperedges in zip(H_train, hedge_train):
        hyp = hypergraph_generation(N, I, hyperedges)
        Is.append(hyp["I"])
        Dv_invs.append(hyp["Dv_inv"])
        De_invs.append(hyp["De_inv"])
        Hs.append(hyp["H"].T)
        Ws.append(hyp["W"])
        i += 1
    Is = torch.stack(Is)
    train_dataset = HyperDataset(Is, Dv_invs, De_invs, Hs, Ws)

    Is = []
    Dv_invs = []
    De_invs = []
    Hs = []
    Ws = []
    i = 0
    # optimals = []
    for I, hyperedges in zip(H_val, hedge_val):
        hyp = hypergraph_generation(N, I, hyperedges)
        Is.append(hyp["I"])
        Dv_invs.append(hyp["Dv_inv"])
        De_invs.append(hyp["De_inv"])
        Hs.append(hyp["H"].T)
        Ws.append(hyp["W"])
        # optimal = exh_solver(N, None, [noise_power]*N, hyp["I"].detach().cpu().numpy(), hyp["H"].detach().cpu().to_dense().numpy().T)[0]
        # optimals.append(optimal)
        i += 1
    # print(f"Optimal values: {np.mean(optimals)}")

    Is = torch.stack(Is)
    test_dataset = HyperDataset(Is, Dv_invs, De_invs, Hs, Ws)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_vec = torch.from_numpy(noise_vec).to(device)
    model = HGNNModel(N).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=0.005, weight_decay=1e-6)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    loss_fn = CustomLossBatch()

    model.eval()
    optimizer.zero_grad()
    accumulated_loss = 0
    accumulated_utility = 0
    tau = 1
    max_iter = 1000
    batch_counter = 0
    with torch.no_grad():
        for i, (X, Dv_inv, De_inv, H, W) in enumerate(train_loader):
            # Move data to device
            X = X.to(device)
            Dv_inv = Dv_inv.to(device)
            De_inv = De_inv.to(device)
            H = H.to(device)
            W = W.to(device)

            # Forward pass
            z = model(X, Dv_inv, De_inv, H, W)

            RHS_const = H.to_dense().T.sum(dim=1) - 1
            LHS_const = H.to_dense().T
            z = z.unsqueeze(0)

            # Apply linsat layer
            z = linsat_layer(z.float(), A=LHS_const.float(), b=RHS_const.float(),
                            tau=tau, max_iter=max_iter, dummy_val=0,
                            no_warning=False, grouped=False).double()

            # Calculate loss and utility
            utility_tr = utility_fn(z, X, noise_vec)
            loss_tr = loss_fn(z, X, noise_vec, gamma=0.0)[0]

            # Scale the loss by accumulation steps
            # loss_tr = loss_tr / accumulation_steps
            # Accumulate loss
            accumulated_loss += loss_tr
            accumulated_utility += utility_tr
            if i % 100 == 0 and i > 0:
                break

        # Evaluate on validation set
        accumulated_utility_val = 0
        accumulated_loss_val = 0
        for i, (X, Dv_inv, De_inv, H, W) in enumerate(test_loader):
            X = X.to(device)
            Dv_inv = Dv_inv.to(device)
            De_inv = De_inv.to(device)
            H = H.to(device)
            W = W.to(device)

            z = model(X, Dv_inv, De_inv, H, W)
            RHS_const = H.to_dense().T.sum(dim=1) - 1
            LHS_const = H.to_dense().T
            z = z.unsqueeze(0)
            z = linsat_layer(z.float(), A=LHS_const.float(), b=RHS_const.float(),
                            tau=tau, max_iter=max_iter, dummy_val=0,
                            no_warning=False, grouped=False).double()
            utility = utility_fn(z, X, noise_vec)
            loss = loss_fn(z, X, noise_vec, gamma=0.0)[0]
            accumulated_utility_val += utility
            accumulated_loss_val += loss
            if i % 100 == 0 and i > 0:
                break
        accumulated_utility, accumulated_utility_val = accumulated_utility/100, accumulated_utility_val/100
        accumulated_loss, accumulated_loss_val = accumulated_loss/100, accumulated_loss_val/100
        print(f"epoch: {0}, batch: {0}, loss_tr: {accumulated_loss:.3f}, "
                f"loss_val: {accumulated_loss_val:.3f}, utility_tr: {accumulated_utility:.3f}, "
                f"utility_val: {accumulated_utility_val:.3f}, z: {None}")

    epochs = 10
    accumulation_steps = 64  # Number of batches to accumulate before backprop

    train_losses = []
    val_losses = []
    train_utilities = []
    val_utilities = []
    best_utility = 0
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()

        batch_counter = 0
        loss_acc = 0
        train_metrics = {'loss': [], 'utility': []}
        for i, (X, Dv_inv, De_inv, H, W) in enumerate(train_loader):
            # Move data to device
            X = X.to(device)
            Dv_inv = Dv_inv.to(device)
            De_inv = De_inv.to(device)
            H = H.to(device)
            W = W.to(device)

            # Forward pass
            z = model(X, Dv_inv, De_inv, H, W)

            RHS_const = H.to_dense().T.sum(dim=1) - 1
            LHS_const = H.to_dense().T
            z = z.unsqueeze(0)

            # Apply linsat layer
            z = linsat_layer(z.float(), A=LHS_const.float(), b=RHS_const.float(),
                            tau=tau, max_iter=max_iter, dummy_val=0,
                            no_warning=False, grouped=False).double()

            # Calculate loss and utility
            utility_tr = utility_fn(z, X, noise_vec)
            loss_tr = loss_fn(z, X, noise_vec, gamma=0.0)[0]
            loss_tr.backward()

            # print(list(model.parameters())[0].grad.norm().item(), X.norm().item(), z.norm().item())

            # Accumulate loss
            train_metrics['loss'].append(loss_tr.detach().cpu().numpy())
            train_metrics['utility'].append(utility_tr.detach().cpu().numpy())

            # Perform backpropagation after accumulating enough batches
            if batch_counter >= accumulation_steps:
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

                # Reset accumulation
                batch_counter = 0

                # Evaluation phase (both training and validation)
                model.eval()
                val_metrics = {'loss': [], 'utility': []}
                zs  = []
                with torch.no_grad():
                    # Evaluate on validation set
                    for (X, Dv_inv, De_inv, H, W) in test_loader:
                        X = X.to(device)
                        Dv_inv = Dv_inv.to(device)
                        De_inv = De_inv.to(device)
                        H = H.to(device)
                        W = W.to(device)

                        z = model(X, Dv_inv, De_inv, H, W)
                        RHS_const = H.to_dense().T.sum(dim=1) - 1
                        LHS_const = H.to_dense().T
                        z = z.unsqueeze(0)
                        z = linsat_layer(z.float(), A=LHS_const.float(), b=RHS_const.float(),
                                        tau=tau, max_iter=max_iter, dummy_val=0,
                                        no_warning=False, grouped=False).double()
                        zs.append(z.cpu().numpy())
                        utility = utility_fn(z, X, noise_vec)
                        loss = loss_fn(z, X, noise_vec, gamma=0.0)[0]
                        val_metrics['loss'].append(loss.cpu().numpy())
                        val_metrics['utility'].append(utility.cpu().numpy())

                # Calculate and store metrics
                train_loss = np.mean(train_metrics['loss'])
                train_utility = np.mean(train_metrics['utility'])
                val_loss = np.mean(val_metrics['loss'])
                val_utility = np.mean(val_metrics['utility'])

                if val_utility > best_utility:
                    best_utility = val_utility
                    torch.save(model.state_dict(), 'model/best_model.pth')


                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_utilities.append(train_utility)
                val_utilities.append(val_utility)

                print(f"epoch: {epoch}, batch: {i}, loss_tr: {train_loss:.3f}, "
                      f"loss_val: {val_loss:.3f}, utility_tr: {train_utility:.3f}, "
                      f"utility_val: {val_utility:.3f}, z: {np.mean(zs)}")
                # Set model back to training mode
                model.train()
            batch_counter += 1

