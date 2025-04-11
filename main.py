import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from LinSATNet import linsat_layer
from networks import HGNNModel
from model import CustomLossBatch, utility_fn
from utils import HyperDataset, get_data, hypergraph_generation, custom_collate_fn

if __name__ == '__main__':
    num_nodes = 10  # Number of nodes (links)
    # E_H = 5   # Number of hyperedges
    # N = 0.1  # Noise power
    N_db = -136.87  # dB
    N = 10**(N_db/10)
    theta = 10  # Thresholds for hyperedges
    train_samples = 100
    test_samples = 20
    xy_lim = 2000
    itens_train, hlist_train, locs_train = get_data(train_samples, num_nodes, N, xy_lim, theta)
    itens_test, hlist_test, locs_test = get_data(test_samples, num_nodes, N, xy_lim, theta)
    hyps_train, hyps_test = [], []

    Is = []
    Dv_invs = []
    De_invs = []
    Hs = []
    Ws = []

    i = 0
    for I, hyperedges in zip(itens_train, hlist_train):
        hyp = hypergraph_generation(num_nodes, I, hyperedges)
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
    for I, hyperedges in zip(itens_test, hlist_test):
        hyp = hypergraph_generation(num_nodes, I, hyperedges)
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
    model = HGNNModel(num_nodes).to(device)

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
                    # Evaluate on training set
                    # for (X, Dv_inv, De_inv, H, W) in train_loader:
                    #     X = X.to(device)
                    #     Dv_inv = Dv_inv.to(device)
                    #     De_inv = De_inv.to(device)
                    #     H = H.to(device)
                    #     W = W.to(device)
                    #
                    #     z = model(X, Dv_inv, De_inv, H, W)
                    #     RHS_const = H.to_dense().T.sum(dim=1) - 1
                    #     LHS_const = H.to_dense().T
                    #     z = z.unsqueeze(0)
                    #     z = linsat_layer(z.float(), A=LHS_const.float(), b=RHS_const.float(),
                    #                     tau=tau, max_iter=max_iter, dummy_val=0,
                    #                     no_warning=False, grouped=False).double()
                    #
                    #     utility = utility_fn(z, X, N)
                    #     loss = loss_fn(z, X, N, gamma=0.0)[0]
                    #     train_metrics['loss'].append(loss.cpu().numpy())
                    #     train_metrics['utility'].append(utility.cpu().numpy())

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
