import torch, math, numpy as np, scipy.sparse as sp
import torch.nn as nn, torch.nn.functional as F, torch.nn.init as init

from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from LinSATNet import linsat_layer



class HyperGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, a, b, reapproximate=True, cuda=True):
        super(HyperGraphConvolution, self).__init__()
        self.a, self.b = a, b
        self.reapproximate, self.cuda = reapproximate, cuda

        self.W = Parameter(torch.FloatTensor(a, b))
        self.bias = Parameter(torch.FloatTensor(b))
        self.reset_parameters()
        


    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)



    def forward(self, structure, H, m=True):
        W, b = self.W, self.bias
        HW = torch.mm(H, W)

        if self.reapproximate:
            n, X = H.shape[0], HW.cpu().detach().numpy()
            A = Laplacian(n, structure, X, m)
        else: A = structure

        if self.cuda: A = A.cuda()
        A = Variable(A)

        AHW = SparseMM.apply(A, HW)
        return AHW + b



    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.a) + ' -> ' \
               + str(self.b) + ')'



class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """
    @staticmethod
    def forward(ctx, M1, M2):
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(ctx, g):
        M1, M2 = ctx.saved_tensors
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g1 = torch.mm(g, M2.t())

        if ctx.needs_input_grad[1]:
            g2 = torch.mm(M1.t(), g)

        return g1, g2



def Laplacian(V, E, X, m):
    """
    approximates the E defined by the E Laplacian with/without mediators

    arguments:
    V: number of vertices
    E: dictionary of hyperedges (key: hyperedge, value: list/set of hypernodes)
    X: features on the vertices
    m: True gives Laplacian with mediators, while False gives without

    A: adjacency matrix of the graph approximation
    returns: 
    updated data with 'graph' as a key and its value the approximated hypergraph 
    """
    
    edges, weights = [], {}
    rv = np.random.rand(X.shape[1])

    for k in E.keys():
        hyperedge = list(E[k])
        
        p = np.dot(X[hyperedge], rv)   #projection onto a random vector rv
        s, i = np.argmax(p), np.argmin(p)
        Se, Ie = hyperedge[s], hyperedge[i]

        # two stars with mediators
        c = 2*len(hyperedge) - 3    # normalisation constant
        if m:
            
            # connect the supremum (Se) with the infimum (Ie)
            edges.extend([[Se, Ie], [Ie, Se]])
            
            if (Se,Ie) not in weights:
                weights[(Se,Ie)] = 0
            weights[(Se,Ie)] += float(1/c)

            if (Ie,Se) not in weights:
                weights[(Ie,Se)] = 0
            weights[(Ie,Se)] += float(1/c)
            
            # connect the supremum (Se) and the infimum (Ie) with each mediator
            for mediator in hyperedge:
                if mediator != Se and mediator != Ie:
                    edges.extend([[Se,mediator], [Ie,mediator], [mediator,Se], [mediator,Ie]])
                    weights = update(Se, Ie, mediator, weights, c)
        else:
            edges.extend([[Se,Ie], [Ie,Se]])
            e = len(hyperedge)
            
            if (Se,Ie) not in weights:
                weights[(Se,Ie)] = 0
            weights[(Se,Ie)] += float(1/e)

            if (Ie,Se) not in weights:
                weights[(Ie,Se)] = 0
            weights[(Ie,Se)] += float(1/e)    
    
    return adjacency(edges, weights, V)



def update(Se, Ie, mediator, weights, c):
    """
    updates the weight on {Se,mediator} and {Ie,mediator}
    """    
    
    if (Se,mediator) not in weights:
        weights[(Se,mediator)] = 0
    weights[(Se,mediator)] += float(1/c)

    if (Ie,mediator) not in weights:
        weights[(Ie,mediator)] = 0
    weights[(Ie,mediator)] += float(1/c)

    if (mediator,Se) not in weights:
        weights[(mediator,Se)] = 0
    weights[(mediator,Se)] += float(1/c)

    if (mediator,Ie) not in weights:
        weights[(mediator,Ie)] = 0
    weights[(mediator,Ie)] += float(1/c)

    return weights



def adjacency(edges, weights, n):
    """
    computes an sparse adjacency matrix

    arguments:
    edges: list of pairs
    weights: dictionary of edge weights (key: tuple representing edge, value: weight on the edge)
    n: number of nodes

    returns: a scipy.sparse adjacency matrix with unit weight self loops for edges with the given weights
    """
    
    dictionary = {tuple(item): index for index, item in enumerate(edges)}
    edges = [list(itm) for itm in dictionary.keys()]   
    organised = []

    for e in edges:
        i,j = e[0],e[1]
        w = weights[(i,j)]
        organised.append(w)

    edges, weights = np.array(edges), np.array(organised)
    adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + sp.eye(n)

    A = symnormalise(sp.csr_matrix(adj, dtype=np.float32))
    A = ssm2tst(A)
    return A



def symnormalise(M):
    """
    symmetrically normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1/2} M D^{-1/2} 
    where D is the diagonal node-degree matrix
    """
    
    d = np.array(M.sum(1))
    
    dhi = np.power(d, -1/2).flatten()
    dhi[np.isinf(dhi)] = 0.
    DHI = sp.diags(dhi)    # D half inverse i.e. D^{-1/2}
    
    return (DHI.dot(M)).dot(DHI) 



def ssm2tst(M):
    """
    converts a scipy sparse matrix (ssm) to a torch sparse tensor (tst)

    arguments:
    M: scipy sparse matrix

    returns:
    a torch sparse tensor of M
    """
    
    M = M.tocoo().astype(np.float32)
    
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)



def gumbel_linsat_layer(scores, A, b,
                        max_iter=100, tau=1., noise_fact=1., sample_num=1000):
    def sample_gumbel(t_like, eps=1e-20):
        """
        randomly sample standard gumbel variables
        """
        u = torch.empty_like(t_like).uniform_()
        return -torch.log(-torch.log(u + eps) + eps)


    s_rep = torch.repeat_interleave(scores.unsqueeze(0), sample_num, dim=0)
    gumbel_noise = sample_gumbel(s_rep) * noise_fact

    s_rep = s_rep + gumbel_noise

    #u = torch.rand(len(scores))
    #a = (u < scores).float()
    #s_rep = a + (scores - scores.detach())
    #s_rep = s_rep.unsqueeze(0)
    




    output = linsat_layer(s_rep, A=A, b=b, tau=0.01, max_iter=400, dummy_val=0)

    return output


def get_hyperedges(V_H, S, N, I, theta=0.5, k=4):
    """
    (4) in thet paper Maximal Scheduling in Wireless Ad Hoc Networks With Hypergraph Interference Models
    """
    idx_sort = np.argsort(I - np.diag(S), axis=1)[:, ::-1][:, :-1]  # Sort in descending order
    hyperedges = []
    for i in range(V_H):
        hyperedge = [i]
        cumulative_intf = 0
        for j in idx_sort[i]:
            hyperedge.append(int(j))
            theta_hat = S[i]/(N+cumulative_intf+I[i, j])
            if theta_hat < theta:
                cumulative_intf += I[i, j]
                if len(hyperedge) == k:
                    break
            else:
                break
        hyperedges.append(hyperedge)
    return hyperedges


def get_data(num_samples, V_H, N=0.1, theta=0.5, k=4):
    itens = []  # interference tensor
    hlist = []  # list of hyperedges
    for _ in range(num_samples):
        # Signal strengths and interference matrix
        I = np.random.rand(V_H, V_H)  # Interference matrix (I_ij)
        S = np.diag(I)        
        # Hyperedges (list of node indices per hyperedge)
        hyperedges = get_hyperedges(V_H, S, N, I, theta, k)
        itens.append(I)
        hlist.append(hyperedges)
    itens = np.array(itens)
    return itens, hlist


def get_H(hlist_train, V_H, train_samples):
    H = np.zeros((V_H*train_samples, V_H), dtype=np.float64)  # [hedge, hnode]
    for hypg_idx, hypg in enumerate(hlist_train):
        for hypedge_idx, hypedge in enumerate(hypg):
            e_idx = hypg_idx*len(hypg) + hypedge_idx
            for hypnode in hypedge:
                H[e_idx, hypnode] = 1
    H = H.reshape(train_samples, V_H, V_H)
    H = H.transpose(0, 2, 1)  # [batch, hnode, hedge]
    return H