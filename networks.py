import torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN, ReLU6 as ReLU6
from torch.autograd import Variable


class HGNNModel_2layer(nn.Module):
    def __init__(self, V_H, hidden_dims=[15, 1]):
        super(HGNNModel_2layer, self).__init__()

        # Define the learnable parameters as nn.Parameter
        self.theta_1 = nn.Parameter(torch.empty(V_H, hidden_dims[0], dtype=torch.float64))
        self.theta_2 = nn.Parameter(torch.empty(hidden_dims[0], hidden_dims[1], dtype=torch.float64))

        # Initialize the parameters with smaller values
        nn.init.xavier_uniform_(self.theta_1, gain=0.1)
        nn.init.xavier_uniform_(self.theta_2, gain=0.1)

        # Add batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dims[0], momentum=0.0, dtype=torch.float64)

    def forward(self, X, Dv_inv, De_inv, H, W):
        # Convert inputs to float64
        X = X.double()
        Dv_inv = Dv_inv.double()
        De_inv = De_inv.double()
        H = H.double()
        W = W.double()

        # Convert 1D tensors to diagonal matrices
        Dv_inv_diag = torch.diag(Dv_inv)
        De_inv_diag = torch.diag(De_inv)
        W_diag = torch.diag(W)

        # First layer
        I_theta1 = torch.matmul(X, self.theta_1)
        Dv_H = torch.matmul(Dv_inv_diag, H)
        W_De = torch.matmul(W_diag, De_inv_diag)
        H_T = H.transpose(0, 1)
        Dv_I_theta1 = torch.matmul(Dv_inv_diag, I_theta1)

        step1 = torch.matmul(Dv_H, W_De)
        step2 = torch.matmul(step1, H_T)
        Xbar1 = torch.matmul(step2, Dv_I_theta1)
        X1 = torch.nn.functional.leaky_relu(Xbar1, negative_slope=0.01)
        X1 = self.bn1(X1)

        # Second layer
        I_theta2 = torch.matmul(X1, self.theta_2)
        Dv_I_theta2 = torch.matmul(Dv_inv_diag, I_theta2)

        step1 = torch.matmul(Dv_H, W_De)
        step2 = torch.matmul(step1, H_T)
        Xbar2 = torch.matmul(step2, Dv_I_theta2)
        X2 = torch.sigmoid(Xbar2)
        X2 = X2.squeeze()

        return X2
    

class HGNNModel(nn.Module):
    def __init__(self, V_H, hidden_dims=[15, 15, 1]):
        super(HGNNModel, self).__init__()

        # Define the learnable parameters as nn.Parameter
        self.theta_1 = nn.Parameter(torch.empty(V_H, hidden_dims[0], dtype=torch.float64))
        self.theta_2 = nn.Parameter(torch.empty(hidden_dims[0], hidden_dims[1], dtype=torch.float64))
        self.theta_3 = nn.Parameter(torch.empty(hidden_dims[1], hidden_dims[2], dtype=torch.float64))

        # Initialize the parameters with smaller values
        nn.init.xavier_uniform_(self.theta_1)
        nn.init.xavier_uniform_(self.theta_2)
        nn.init.xavier_uniform_(self.theta_3)

        # Add batch normalization layers
        self.bn1 = nn.LayerNorm(hidden_dims[0], dtype=torch.float64)
        self.bn2 = nn.LayerNorm(hidden_dims[1], dtype=torch.float64)

    def forward(self, X, Dv_inv, De_inv, H, W):
        # Convert inputs to float64
        X = X.double()
        Dv_inv = Dv_inv.double()
        De_inv = De_inv.double()
        H = H.double()
        W = W.double()

        # Convert 1D tensors to diagonal matrices
        Dv_inv_diag = torch.diag(Dv_inv)
        De_inv_diag = torch.diag(De_inv)
        W_diag = torch.diag(W)

        # First layer
        I_theta1 = torch.matmul(X, self.theta_1)
        Dv_H = torch.matmul(Dv_inv_diag, H)
        W_De = torch.matmul(W_diag, De_inv_diag)
        H_T = H.transpose(0, 1)
        Dv_I_theta1 = torch.matmul(Dv_inv_diag, I_theta1)

        step1 = torch.matmul(Dv_H, W_De)
        step2 = torch.matmul(step1, H_T)
        Xbar1 = torch.matmul(step2, Dv_I_theta1)
        X1 = torch.nn.functional.leaky_relu(Xbar1, negative_slope=0.01)
        X1 = self.bn1(X1)

        # Second layer
        I_theta2 = torch.matmul(X1, self.theta_2)
        Dv_I_theta2 = torch.matmul(Dv_inv_diag, I_theta2)

        step1 = torch.matmul(Dv_H, W_De)
        step2 = torch.matmul(step1, H_T)
        Xbar2 = torch.matmul(step2, Dv_I_theta2)
        X2 = torch.nn.functional.leaky_relu(Xbar2, negative_slope=0.01)
        X2 = self.bn2(X2)

        # Third layer
        I_theta3 = torch.matmul(X2, self.theta_3)
        Dv_I_theta3 = torch.matmul(Dv_inv_diag, I_theta3)

        step1 = torch.matmul(Dv_H, W_De)
        step2 = torch.matmul(step1, H_T)
        Xbar3 = torch.matmul(step2, Dv_I_theta3)
        X3 = torch.sigmoid(Xbar3)
        X3 = X3.squeeze()

        return X3


class MLP(nn.Module):
    def __init__(self, V_H):
        super(MLP, self).__init__()
        self.layer1 = Seq(Lin(V_H*V_H, 1024, dtype=torch.float64), ReLU())
        self.layer2 = Seq(Lin(1024, 512, dtype=torch.float64), ReLU())
        self.layer3 = Seq(Lin(512, 128, dtype=torch.float64), ReLU())
        self.layer4 = Seq(Lin(128, V_H, dtype=torch.float64), Sigmoid())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
