import networks
import torch, os, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import utils
import schedulefree



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
    numerator = zs*torch.diag(X)
    X_bar = X - torch.diag(torch.diag(X))
    Xz = X_bar @ zs
    denumerator = var_noise + Xz
    u = torch.sum(numerator/denumerator)
    return u
    

def train(HyperGCN, dataset, epochs, batch_size=10):
    """
    train for a certain number of epochs

    arguments:
	HyperGCN: a dictionary containing model details (gcn, optimiser)
	dataset: the entire dataset
	epochs: number of training epochs

	returns:
	the trained model
    """    
    
    hypergcn, optimiser = HyperGCN['model'], HyperGCN['optimiser']
    hypergcn.train()
    optimiser.train()

    loss_fn = CustomLoss()
    
    # X, H = dataset['I'], dataset["H"]
    itens, hlist = dataset['I'], dataset['H']

    # RHS_const = H.T.sum(dim=1) - 1
    # LHS_const = H.T
    utility = []
    for epoch in tqdm(range(epochs)):
        loss_batch, u_batch = [], []
        i = 0
        optimiser.zero_grad()
        for X, H in zip(itens, hlist):
            Z = hypergcn(X)  # tilde
    
            # constrained_output = utils.gumbel_linsat_layer(Z, LHS_const, RHS_const)  # Zs
    
            # print(torch.mean(constrained_output, dim=0))  # Z = Z_mean
    
            # loss = loss_fn(constrained_output, X, 0.01)
            loss, u = loss_fn(Z, X, 0.1)
            loss_batch.append(loss)
            u_batch.append(u)
            if i < batch_size:
                i+=1
                continue
            i = 0
            np.mean(loss_batch).backward()
            optimiser.step()
            optimiser.zero_grad()
            utility.append(np.mean(u_batch))
            loss_batch, u_batch = [], []
        
        # print utility per epoch
        print("Epoch: {0}, Utility: {1}".format(epoch, u))


    HyperGCN['model'] = hypergcn
    HyperGCN['utility'] = utility
    return HyperGCN




def initialise(dataset):
    """
    initialises GCN, optimiser, and features, and set GPU 
    
    arguments:
    dataset: the entire dataset (with graph, features, labels as keys)
    
    returns:
    a dictionary with model details (hypergcn, optimiser)    
    """
    
    HyperGCN = {}
    V, E = dataset['n'], dataset['E']
    X = dataset['I']

    # hypergcn and optimiser
    d = X.shape[1]
    c = V
    hypergcn = networks.HyperGCN(V, E, X, d, c, False, False)
    #optimiser = optim.Adam(hypergcn.parameters(), lr=0.0001)
    optimiser = schedulefree.AdamWScheduleFree(hypergcn.parameters(), lr=0.01)


    # node features in sparse representation
    X = sp.csr_matrix(np.array(X), dtype=np.float32)
    X = torch.FloatTensor(np.array(X.todense()))
    


    # cuda
    Cuda = torch.cuda.is_available()
    if Cuda:
        hypergcn.cuda()
        X = X.cuda()

    # update dataset with torch autograd variable
    dataset['I'] = Variable(X)

    # update model and optimiser
    HyperGCN['model'] = hypergcn
    HyperGCN['optimiser'] = optimiser
    return HyperGCN


