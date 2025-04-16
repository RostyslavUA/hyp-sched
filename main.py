import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from LinSATNet import linsat_layer
from networks import HGNNModel
from model import CustomLossBatch, utility_fn
from utils import HyperDataset, get_data, hypergraph_generation, custom_collate_fn
from conflict_vs_hypergraph import generate_channel_matrix, build_hyperedges, build_conflict_edges


def data_generate(train_size):
    H_train, hedge_train = [], []
    for _ in range(train_size):
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
    train_size = 100
    val_size = 10

    # Noise vector for all links
    noise_vec = np.full(N, noise_power)
    H_train, hedge_train = data_generate(train_size)
    H_val, hedge_val = data_generate(val_size)
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
    for I, hyperedges in zip(H_val, hedge_val):
        hyp = hypergraph_generation(N, I, hyperedges)
        Is.append(hyp["I"])
        Dv_invs.append(hyp["Dv_inv"])
        De_invs.append(hyp["De_inv"])
        Hs.append(hyp["H"].T)
        Ws.append(hyp["W"])
        i += 1
    Is = torch.stack(Is)
    test_dataset = HyperDataset(Is, Dv_invs, De_invs, Hs, Ws)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    epochs = 10
    accumulation_steps = 20  # Number of batches to accumulate before backprop

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HGNNModel(N).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    # optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=0.005, weight_decay=1e-6)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    loss_fn = CustomLossBatch()

    train_losses = []
    val_losses = []
    train_utilities = []
    val_utilities = []
    tau = 1
    max_iter = 1000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        accumulated_loss = 0
        accumulated_utility = 0
        batch_counter = 0

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
            utility_tr = utility_fn(z, X, N)
            loss_tr = loss_fn(z, X, N, gamma=0.0)[0]

            # Add L2 regularization
            l2_reg = 0.001 * sum(p.pow(2.0).sum() for p in model.parameters())
            loss_tr = loss_tr + l2_reg

            # Scale the loss by accumulation steps
            loss_tr = loss_tr / accumulation_steps
            utility_tr = utility_tr / accumulation_steps

            # Accumulate loss
            accumulated_loss += loss_tr
            accumulated_utility += utility_tr
            batch_counter += 1

            # Perform backpropagation after accumulating enough batches
            if batch_counter == accumulation_steps:
                # Backward pass
                accumulated_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

                # Reset accumulation
                batch_counter = 0

                # Evaluation phase (both training and validation)
                model.eval()
                val_metrics = {'loss': [], 'utility': []}

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

                        utility = utility_fn(z, X, N)
                        loss = loss_fn(z, X, N, gamma=0.0)[0]
                        val_metrics['loss'].append(loss.cpu().numpy())
                        val_metrics['utility'].append(utility.cpu().numpy())

                # Calculate and store metrics
                train_loss = accumulated_loss
                train_utility = accumulated_utility
                val_loss = np.mean(val_metrics['loss'])
                val_utility = np.mean(val_metrics['utility'])

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_utilities.append(train_utility)
                val_utilities.append(val_utility)

                print(f"epoch: {epoch}, batch: {i}, loss_tr: {train_loss:.3f}, "
                      f"loss_val: {val_loss:.3f}, utility_tr: {train_utility:.3f}, "
                      f"utility_val: {val_utility:.3f}")
                accumulated_loss = 0
                # Set model back to training mode
                model.train()
