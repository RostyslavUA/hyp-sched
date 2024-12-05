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

    def forward(self, zs, X, var_noise):
        loss = 0
        for i in range(len(zs)):
            z = zs[i]
            all_info = torch.einsum('j,ij->ij', z, X)
            numerators = torch.diag(all_info)
            denumerators = torch.sum(all_info, dim=1) - numerators
            loss += -torch.sum(numerators/(var_noise + denumerators))
        return loss / len(zs)
    

def train(HyperGCN, dataset, epochs):
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

    loss_fn = CustomLoss()
    
    X, H = dataset['I'], dataset["H"]

    RHS_const = H.T.sum(dim=1) - 1
    LHS_const = H.T

    for epoch in tqdm(range(epochs)):

        optimiser.zero_grad()
        Z = hypergcn(X)

        constrained_output = utils.gumbel_linsat_layer(Z, LHS_const, RHS_const)

        print(torch.mean(constrained_output, dim=0))

        loss = loss_fn(constrained_output, X, 0.01)
        loss.backward()
        optimiser.step()
        
        # print loss per epoch
        print("Epoch: {0}, Loss: {1}".format(epoch, loss))


    HyperGCN['model'] = hypergcn
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
    optimiser = schedulefree.AdamWScheduleFree(hypergcn.parameters(), lr=0.001)


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


