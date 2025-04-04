import networks
import torch, os, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import utils
import schedulefree
from linsat import linsat_layer_modified
from utils import check_feasibility, gumbel_linsat_layer
from LinSATNet import linsat_layer




class CustomLossBatch(nn.Module):
    def __init__(self):
        super(CustomLossBatch, self).__init__()

    def forward(self, zs, X, var_noise, gamma=0.0):
        all_info = zs.unsqueeze(1) * X
        numerators = all_info.diagonal(dim1=1, dim2=2)  
        denominators = torch.sum(all_info, dim=2) - numerators
        fraction = numerators / (var_noise + denominators)
        fraction = torch.log2(1+fraction)
        utility = torch.sum(fraction, dim=1) 
        loss = -torch.sum(utility) + gamma*torch.sum(torch.linalg.norm(fraction, ord=1, dim=1))

        return loss, torch.sum(utility)
        

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, zs, X, var_noise, gamma=0.0):
        loss = 0
        numerator = zs*torch.diag(X)
        X_bar = X - torch.diag(torch.diag(X))
        Xz = X_bar @ zs
        denumerator = var_noise + Xz
        sinr = numerator/denumerator
        u = torch.sum(sinr)
        loss = -u  + gamma*torch.linalg.norm(sinr, ord=1)
        return loss, u


def utility_fn(zs, X, var_noise):
    zs = torch.round(zs)
    all_info = zs.unsqueeze(1) * X
    numerators = all_info.diagonal(dim1=1, dim2=2)  
    denominators = torch.sum(all_info, dim=2) - numerators
    fraction = numerators / (var_noise + denominators)
    fraction = torch.log2(1+fraction)
    utility = torch.sum(fraction) 
    return utility
    

def train_handler(model, hgnn_weights, hgnn_batchnorm, loss_fn, optimizer, train_loader, test_loader, epochs, model_name, tau_linsat, iter_linsat, gumbel_flag, gumble_samples, N, device):

    train_utility, test_utility = [], []
    loss_train, loss_test = [], []
    batch_size = train_loader.batch_size

    ################### Evaluation without training ###################
    if model_name == "HGNN":
            hgnn_batchnorm = [b.eval()  for b in hgnn_batchnorm]  # Set batch norm in training mode
    else:
        # MLP
        model.eval()

    with torch.no_grad():
        ####### on Train data #######
        utility_epoch = []
        loss_train_epoch = []
        for (X, Dv_inv, De_inv, H, W) in train_loader:
            X, H, W = X.to(device), H.to(device), W.to(device)
            Dv_inv, De_inv = Dv_inv.to(device), De_inv.to(device)
            if model_name == "HGNN":
                z = model(X, Dv_inv, De_inv, H, W, hgnn_weights, hgnn_batchnorm)
            else:
                # MLP
                V_H = X.shape[2]
                z = model(X.view(-1, V_H*V_H))

            # Linsat
            RHS_const = H.transpose(2, 1).sum(dim=2) - 1
            LHS_const = H.transpose(2, 1)
            if gumbel_flag == False:
                z = linsat_layer_modified(z.float(), A=LHS_const.float(), b=RHS_const.float(), tau=tau_linsat, max_iter=iter_linsat, dummy_val=0, no_warning=False, grouped=False).double()
                loss = loss_fn(z, X, N, gamma=0.0)[0]
                utility = utility_fn(z, X, N)  # utility based on discrete values of z
            else:
                z = gumbel_linsat_layer(z, A=LHS_const.float(), b=RHS_const.float(), tau=tau_linsat, max_iter=iter_linsat, noise_fact=1., sample_num=gumble_samples)
                loss = 0
                for i in range(gumble_samples):
                    loss += loss_fn(z[i], X, N, gamma=0.0)[0]
                loss /= gumble_samples

                utility = 0
                for i in range(gumble_samples):
                    utility += utility_fn(z[i], X, N)
                utility /= gumble_samples

            utility_epoch.append(utility.item()/batch_size)
            loss_train_epoch.append(loss.item()/batch_size)
        utility_epoch_mean = np.mean(utility_epoch)
        train_utility.append(utility_epoch_mean)
        loss_train.append(np.mean(loss_train_epoch))
        print(f"Initial utility per hypergraph in Training: {train_utility[-1]}")

        ####### on Test data #######
        utility_epoch_test = []
        loss_test_epoch = []
        for (X, Dv_inv, De_inv, H, W) in test_loader:
            X, H, W = X.to(device), H.to(device), W.to(device)
            Dv_inv, De_inv = Dv_inv.to(device), De_inv.to(device)
            if model_name == "HGNN":
                z = model(X, Dv_inv, De_inv, H, W, hgnn_weights, hgnn_batchnorm)
            else:
                # MLP
                V_H = X.shape[2]
                z = model(X.view(-1, V_H*V_H))

            # Linsat 
            RHS_const = H.transpose(2, 1).sum(dim=2) - 1
            LHS_const = H.transpose(2, 1)
            if gumbel_flag == False:
                z = linsat_layer_modified(z.float(), A=LHS_const.float(), b=RHS_const.float(), tau=tau_linsat, max_iter=iter_linsat, dummy_val=0, no_warning=False, grouped=False).double()
                loss = loss_fn(z, X, N, gamma=0.0)[0]
                utility = utility_fn(z, X, N)  # utility based on discrete values of z
            else:
                z = gumbel_linsat_layer(z, A=LHS_const.float(), b=RHS_const.float(), tau=tau_linsat, max_iter=iter_linsat, noise_fact=1., sample_num=gumble_samples)
                loss = 0
                for i in range(gumble_samples):
                    loss += loss_fn(z[i], X, N, gamma=0.0)[0]
                loss /= gumble_samples

                utility = 0
                for i in range(gumble_samples):
                    utility += utility_fn(z[i], X, N)
                utility /= gumble_samples
            utility_epoch_test.append(utility.item()/batch_size)
            loss_test_epoch.append(loss.item()/batch_size)
        utility_epoch_test_mean = np.mean(utility_epoch_test)
        test_utility.append(utility_epoch_test_mean)
        loss_test.append(np.mean(loss_test_epoch))
        print(f"Initial utility per hypergraph in Testing: {test_utility[-1]}")
    



    ################### Traing loop ###################
    optimizer.zero_grad()
    for epoch in range(epochs):
        # Train Phase
        if model_name == "HGNN":
            hgnn_batchnorm = [b.train() for b in hgnn_batchnorm]  # Set batch norm in training mode
        else:
            # MLP
            model.train()
        loss_epoch, utility_epoch, feasibility_epoch = [], [], []
        accumulated_loss = 0.0
        for (X, Dv_inv, De_inv, H, W) in train_loader:
            X, H, W = X.to(device), H.to(device), W.to(device)
            Dv_inv, De_inv = Dv_inv.to(device), De_inv.to(device)

            if model_name == "HGNN":
                z = model(X, Dv_inv, De_inv, H, W, hgnn_weights, hgnn_batchnorm)
            else:
                # MLP
                V_H = X.shape[2]
                z = model(X.view(-1, V_H*V_H))
            # Linsat
            RHS_const = H.transpose(2, 1).sum(dim=2) - 1
            LHS_const = H.transpose(2, 1)
            if gumbel_flag == False:
                z = linsat_layer_modified(z.float(), A=LHS_const.float(), b=RHS_const.float(), tau=tau_linsat, max_iter=iter_linsat, dummy_val=0, no_warning=False, grouped=False).double()
                loss = loss_fn(z, X, N, gamma=0.0)[0]
                utility = utility_fn(z, X, N)  # utility based on discrete values of z
            else:
                z = gumbel_linsat_layer(z, A=LHS_const.float(), b=RHS_const.float(), tau=tau_linsat, max_iter=iter_linsat, noise_fact=1., sample_num=gumble_samples)
                loss = 0
                for i in range(gumble_samples):
                    loss += loss_fn(z[i], X, N, gamma=0.0)[0]
                loss /= gumble_samples

                utility = 0
                for i in range(gumble_samples):
                    utility += utility_fn(z[i], X, N)
                utility /= gumble_samples

            feasibility = check_feasibility(H, z)
            utility_epoch.append(utility.item()/batch_size)
            feasibility_epoch.append(feasibility)
            loss.backward()
            # print(torch.linalg.matrix_norm(theta_HGNN[0].grad))
            optimizer.step()
            optimizer.zero_grad()
            loss_epoch.append(loss.item()/batch_size)
        # Test Phase
        if model_name == "HGNN":
            hgnn_batchnorm = [b.eval()  for b in hgnn_batchnorm]  # Set batch norm in training mode
        else:
            # MLP
            model.eval()
        loss_epoch_test, utility_epoch_test, feasibility_epoch_test = [], [], []
        with torch.no_grad():
            for (X, Dv_inv, De_inv, H, W) in test_loader:
                X, H, W = X.to(device), H.to(device), W.to(device)
                Dv_inv, De_inv = Dv_inv.to(device), De_inv.to(device)
                
                if model_name == "HGNN":
                    z = model(X, Dv_inv, De_inv, H, W, hgnn_weights, hgnn_batchnorm)
                else:
                    # MLP
                    V_H = X.shape[2]
                    z = model(X.view(-1, V_H*V_H))

                # Linsat 
                RHS_const = H.transpose(2, 1).sum(dim=2) - 1
                LHS_const = H.transpose(2, 1)
                if gumbel_flag == False:
                    z = linsat_layer_modified(z.float(), A=LHS_const.float(), b=RHS_const.float(), tau=tau_linsat, max_iter=iter_linsat, dummy_val=0, no_warning=False, grouped=False).double()
                    loss = loss_fn(z, X, N, gamma=0.0)[0]
                    utility = utility_fn(z, X, N)  # utility based on discrete values of z
                else:
                    z = gumbel_linsat_layer(z, A=LHS_const.float(), b=RHS_const.float(), tau=tau_linsat, max_iter=iter_linsat, noise_fact=1., sample_num=gumble_samples)
                    loss = 0
                    for i in range(gumble_samples):
                        loss += loss_fn(z[i], X, N, gamma=0.0)[0]
                    loss /= gumble_samples

                    utility = 0
                    for i in range(gumble_samples):
                        utility += utility_fn(z[i], X, N)
                    utility /= gumble_samples
                feasibility = check_feasibility(H, z)
                utility_epoch_test.append(utility.item()/batch_size)
                feasibility_epoch_test.append(feasibility)
                loss_epoch_test.append(loss.item()/batch_size)
        utility_epoch_mean = np.mean(utility_epoch)
        utility_epoch_test_mean = np.mean(utility_epoch_test)
        feasibility_epoch_mean = np.mean(feasibility_epoch)
        feasibility_epoch_test_mean = np.mean(feasibility_epoch_test)
        loss_epoch_mean = np.mean(loss_epoch)
        loss_test_mean = np.mean(loss_epoch_test)
        
        if epoch % 1 == 0:
            print(f"Epoch: {epoch}, Loss:{np.mean(loss_epoch):.5f} | Train Utility: {utility_epoch_mean:.5f}, Feasibility: {feasibility_epoch_mean:.2f}" \
            f" | Test Utility: {utility_epoch_test_mean:.5f}, Feasibility: {feasibility_epoch_test_mean:.2f}")
        train_utility.append(utility_epoch_mean)
        test_utility.append(utility_epoch_test_mean)
        loss_train.append(loss_epoch_mean)
        loss_test.append(loss_test_mean)
    print(f"Final utility per hypergraph: {utility_epoch_test}")

    return train_utility, test_utility, loss_train, loss_test


def train_handler_batch(model, hgnn_weights, hgnn_batchnorm, loss_fn, optimizer, train_loader, test_loader, epochs, model_name, tau_linsat, iter_linsat, gumbel_flag, gumble_samples, N, batch_sample, device):

    train_utility, test_utility = [], []
    loss_train, loss_test = [], []
    batch_size = train_loader.batch_size

    ################### Evaluation without training ###################
    if model_name == "HGNN":
            hgnn_batchnorm = [b.eval()  for b in hgnn_batchnorm]  # Set batch norm in training mode
    else:
        # MLP
        model.eval()

    with torch.no_grad():
        ####### on Train data #######
        utility_epoch = []
        loss_train_epoch = []
        for (X, Dv_inv, De_inv, H, W) in train_loader:
            X, H, W = X.to(device), H.to(device), W.to(device)
            Dv_inv, De_inv = Dv_inv.to(device), De_inv.to(device)

            if model_name == "HGNN":
                z = model(X, Dv_inv, De_inv, H, W, hgnn_weights, hgnn_batchnorm)
            else:
                # MLP
                V_H = X.shape[2]
                z = model(X.view(-1, V_H*V_H))

            # Linsat
            RHS_const = H.to_dense().squeeze(0).T.sum(dim=1) - 1
            LHS_const = H.to_dense().squeeze(0).T
            z = z.unsqueeze(0)
            if gumbel_flag == False:
                z = linsat_layer(z.float(), A=LHS_const.float(), b=RHS_const.float(), tau=tau_linsat, max_iter=iter_linsat, dummy_val=0, no_warning=False, grouped=False).double()
                #z = linsat_layer_modified(z.float(), A=LHS_const.float(), b=RHS_const.float(), tau=tau_linsat, max_iter=iter_linsat, dummy_val=0, no_warning=False, grouped=False).double()
                loss = loss_fn(z, X, N, gamma=0.0)[0]
                utility = utility_fn(z, X, N)  # utility based on discrete values of z
            else:
                z = gumbel_linsat_layer(z, A=LHS_const.float(), b=RHS_const.float(), tau=tau_linsat, max_iter=iter_linsat, noise_fact=1., sample_num=gumble_samples)
                loss = 0
                for i in range(gumble_samples):
                    loss += loss_fn(z[i], X, N, gamma=0.0)[0]
                loss /= gumble_samples

                utility = 0
                for i in range(gumble_samples):
                    utility += utility_fn(z[i], X, N)
                utility /= gumble_samples

            utility_epoch.append(utility.item()/batch_size)
            loss_train_epoch.append(loss.item()/batch_size)
        utility_epoch_mean = np.mean(utility_epoch)
        train_utility.append(utility_epoch_mean)
        loss_train.append(np.mean(loss_train_epoch))
        print(f"Initial utility per hypergraph in Training: {train_utility[-1]}")

        ####### on Test data #######
        utility_epoch_test = []
        loss_test_epoch = []
        for (X, Dv_inv, De_inv, H, W) in test_loader:
            X, H, W = X.to(device), H.to(device), W.to(device)
            Dv_inv, De_inv = Dv_inv.to(device), De_inv.to(device)

            if model_name == "HGNN":
                z = model(X, Dv_inv, De_inv, H, W, hgnn_weights, hgnn_batchnorm)
            else:
                # MLP
                V_H = X.shape[2]
                z = model(X.view(-1, V_H*V_H))

            # Linsat
            RHS_const = H.to_dense().squeeze(0).T.sum(dim=1) - 1
            LHS_const = H.to_dense().squeeze(0).T
            z = z.unsqueeze(0)
            if gumbel_flag == False:
                z = linsat_layer(z.float(), A=LHS_const.float(), b=RHS_const.float(), tau=tau_linsat, max_iter=iter_linsat, dummy_val=0, no_warning=False, grouped=False).double()
                #z = linsat_layer_modified(z.float(), A=LHS_const.float(), b=RHS_const.float(), tau=tau_linsat, max_iter=iter_linsat, dummy_val=0, no_warning=False, grouped=False).double()
                loss = loss_fn(z, X, N, gamma=0.0)[0]
                utility = utility_fn(z, X, N)  # utility based on discrete values of z
            else:
                z = gumbel_linsat_layer(z, A=LHS_const.float(), b=RHS_const.float(), tau=tau_linsat, max_iter=iter_linsat, noise_fact=1., sample_num=gumble_samples)
                loss = 0
                for i in range(gumble_samples):
                    loss += loss_fn(z[i], X, N, gamma=0.0)[0]
                loss /= gumble_samples

                utility = 0
                for i in range(gumble_samples):
                    utility += utility_fn(z[i], X, N)
                utility /= gumble_samples
            utility_epoch_test.append(utility.item()/batch_size)
            loss_test_epoch.append(loss.item()/batch_size)
        utility_epoch_test_mean = np.mean(utility_epoch_test)
        test_utility.append(utility_epoch_test_mean)
        loss_test.append(np.mean(loss_test_epoch))
        print(f"Initial utility per hypergraph in Testing: {test_utility[-1]}")
    



    ################### Traing loop ###################
    optimizer.zero_grad()
    for epoch in range(epochs):
        # Train Phase
        if model_name == "HGNN":
            hgnn_batchnorm = [b.train() for b in hgnn_batchnorm]  # Set batch norm in training mode
        else:
            # MLP
            model.train()
        loss_epoch, utility_epoch, feasibility_epoch = [], [], []
        accumulated_loss = 0.0
        counter = 0
        for (X, Dv_inv, De_inv, H, W) in train_loader:
            X, H, W = X.to(device), H.to(device), W.to(device)
            Dv_inv, De_inv = Dv_inv.to(device), De_inv.to(device)

            if model_name == "HGNN":
                z = model(X, Dv_inv, De_inv, H, W, hgnn_weights, hgnn_batchnorm)
            else:
                # MLP
                V_H = X.shape[2]
                z = model(X.view(-1, V_H*V_H))

            # Linsat
            RHS_const = H.to_dense().squeeze(0).T.sum(dim=1) - 1
            LHS_const = H.to_dense().squeeze(0).T
            z = z.unsqueeze(0)
            if gumbel_flag == False:
                z = linsat_layer(z.float(), A=LHS_const.float(), b=RHS_const.float(), tau=tau_linsat, max_iter=iter_linsat, dummy_val=0, no_warning=False, grouped=False).double()
                loss = loss_fn(z, X, N, gamma=0.0)[0]
                utility = utility_fn(z, X, N)  # utility based on discrete values of z
            else:
                z = gumbel_linsat_layer(z, A=LHS_const.float(), b=RHS_const.float(), tau=tau_linsat, max_iter=iter_linsat, noise_fact=1., sample_num=gumble_samples)
                loss = 0
                for i in range(gumble_samples):
                    loss += loss_fn(z[i], X, N, gamma=0.0)[0]
                loss /= gumble_samples

                utility = 0
                for i in range(gumble_samples):
                    utility += utility_fn(z[i], X, N)
                utility /= gumble_samples

            feasibility = check_feasibility(H, z)
            utility_epoch.append(utility.item()/batch_size)
            feasibility_epoch.append(feasibility)
            accumulated_loss += loss
            counter += 1
            if counter == batch_sample:
                loss = accumulated_loss / batch_sample
                
                loss.backward()
                # print(torch.linalg.matrix_norm(theta_HGNN[0].grad))
                optimizer.step()
                optimizer.zero_grad()
                loss_epoch.append(loss.item())
                accumulated_loss = 0.0
                counter = 0
        # Test Phase
        if model_name == "HGNN":
            hgnn_batchnorm = [b.eval()  for b in hgnn_batchnorm]  # Set batch norm in training mode
        else:
            # MLP
            model.eval()
        loss_epoch_test, utility_epoch_test, feasibility_epoch_test = [], [], []
        with torch.no_grad():
            for (X, Dv_inv, De_inv, H, W) in test_loader:
                X, H, W = X.to(device), H.to(device), W.to(device)
                Dv_inv, De_inv = Dv_inv.to(device), De_inv.to(device)

                if model_name == "HGNN":
                    z = model(X, Dv_inv, De_inv, H, W, hgnn_weights, hgnn_batchnorm)
                else:
                    # MLP
                    V_H = X.shape[2]
                    z = model(X.view(-1, V_H*V_H))

                # Linsat
                RHS_const = H.to_dense().squeeze(0).T.sum(dim=1) - 1
                LHS_const = H.to_dense().squeeze(0).T
                z = z.unsqueeze(0)
                if gumbel_flag == False:
                    z = linsat_layer(z.float(), A=LHS_const.float(), b=RHS_const.float(), tau=tau_linsat, max_iter=iter_linsat, dummy_val=0, no_warning=False, grouped=False).double()
                    loss = loss_fn(z, X, N, gamma=0.0)[0]
                    utility = utility_fn(z, X, N)  # utility based on discrete values of z
                else:
                    z = gumbel_linsat_layer(z, A=LHS_const.float(), b=RHS_const.float(), tau=tau_linsat, max_iter=iter_linsat, noise_fact=1., sample_num=gumble_samples)
                    loss = 0
                    for i in range(gumble_samples):
                        loss += loss_fn(z[i], X, N, gamma=0.0)[0]
                    loss /= gumble_samples

                    utility = 0
                    for i in range(gumble_samples):
                        utility += utility_fn(z[i], X, N)
                    utility /= gumble_samples
                feasibility = check_feasibility(H, z)
                utility_epoch_test.append(utility.item()/batch_size)
                feasibility_epoch_test.append(feasibility)
                loss_epoch_test.append(loss.item()/batch_size)
        utility_epoch_mean = np.mean(utility_epoch)
        utility_epoch_test_mean = np.mean(utility_epoch_test)
        feasibility_epoch_mean = np.mean(feasibility_epoch)
        feasibility_epoch_test_mean = np.mean(feasibility_epoch_test)
        loss_epoch_mean = np.mean(loss_epoch)
        loss_test_mean = np.mean(loss_epoch_test)
        
        if epoch % 1 == 0:
            print(f"Epoch: {epoch}, Loss:{np.mean(loss_epoch):.5f} | Train Utility: {utility_epoch_mean:.5f}, Feasibility: {feasibility_epoch_mean:.2f}" \
            f" | Test Utility: {utility_epoch_test_mean:.5f}, Feasibility: {feasibility_epoch_test_mean:.2f}")
        train_utility.append(utility_epoch_mean)
        test_utility.append(utility_epoch_test_mean)
        loss_train.append(loss_epoch_mean)
        loss_test.append(loss_test_mean)
    print(f"Final utility per hypergraph: {utility_epoch_test}")

    return train_utility, test_utility, loss_train, loss_test



