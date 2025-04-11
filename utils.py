import torch, math, numpy as np, scipy.sparse as sp
import torch.nn as nn, torch.nn.functional as F, torch.nn.init as init

from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from datagen import generate_data
from linsat import linsat_layer_modified

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
# from shapely.geometry import MultiPoint
from matplotlib.patches import Polygon as MplPolygon
from torch.utils.data import Dataset


def create_sparse_tensor(hyperedges, V_H):
    row_indices = []
    col_indices = []
    values = []

    for i, hyperedge in enumerate(hyperedges):
        for node in hyperedge:
            row_indices.append(i)  # Hyperedge index
            col_indices.append(node)  # Node index
            values.append(1)  # Assuming all values are 1

    # Convert to tensors
    indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
    values = torch.tensor(values, dtype=torch.float32)
    shape = (len(hyperedges), V_H)  # Determine shape

    return torch.sparse_coo_tensor(indices, values, size=shape, dtype=torch.float64)


def hypergraph_generation(V_H, I, hyperedges):
    hypergraph = {}
    hypergraph["I"] = torch.DoubleTensor(I)
    H = create_sparse_tensor(hyperedges, V_H)
    Dv_inv = (H.sum(0)**(-1/2)).to_dense()  # Not diagonal; We will use element-wise multiplication to save memory
    De_inv = (H.sum(1)**(-1)).to_dense()
    W = torch.ones(De_inv.shape, dtype=torch.float64)
    hypergraph["H"] = H
    hypergraph["Dv_inv"] = Dv_inv
    hypergraph["De_inv"] = De_inv
    hypergraph["W"] = W
    return hypergraph


class HyperDataset(Dataset):
    def __init__(self, I, Dv_inv, De_inv, H, W):
        self.I = I
        self.Dv_inv = Dv_inv
        self.De_inv = De_inv
        self.H = H
        self.W = W

    def __len__(self):
        return len(self.I)

    def __getitem__(self, idx):
        return self.I[idx], self.Dv_inv[idx], self.De_inv[idx], self.H[idx], self.W[idx]


def custom_collate_fn(batch):
    return batch[0]


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


    outputs = torch.zeros_like(s_rep)
    for i in range(sample_num):
        outputs[i] = linsat_layer_modified(s_rep[i].float(), A=A, b=b, tau=tau, max_iter=max_iter, dummy_val=0, no_warning=False, grouped=False).double()

    return outputs


def get_hyperedges(V_H, S, N, I, theta=0.5):
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
            if theta_hat > theta:
                cumulative_intf += I[i, j]
            else:
                hyperedges.append(hyperedge)
                hyperedge = [i]
                cumulative_intf = 0
    return hyperedges


def get_data(num_samples, V_H, N=0.1, xy_lim=500, theta=0.5):
    itens = []  # interference tensor
    hlist = []  # list of hyperedges
    locs = []
    for _ in range(num_samples):
        # Signal strengths and interference matrix
        # I = np.random.rand(V_H, V_H)  # Interference matrix (I_ij)
        data_dict = generate_data(tr_iter=1, te_iter=0, batch_size=1, layout='circle', xy_lim=xy_lim, alpha=1/np.sqrt(2), nNodes=V_H, threshold=False, fading=False)
        I = data_dict['train_H'][0][0]
        I = I.T  # [receiver, transmitter]
        S = np.diag(I)     
        # Hyperedges (list of node indices per hyperedge)
        hyperedges = get_hyperedges(V_H, S, N, I, theta)
        itens.append(I)
        hlist.append(hyperedges)
        locs.append(data_dict['tr_locs'])
    itens = np.array(itens)
    return itens, hlist, locs


def get_H(hlist_train, train_samples):
    hlist_train_coo = []
    for b, hlist in enumerate(hlist_train):
        hlist_coo = []
        for hedge in hlist:
            if len(hedge) == 2:
                hlist_coo.append([b] + hedge)
            else:
                for i, hnode in enumerate(hedge):
                    if i == 0:
                        rx = hnode
                    else:
                        hlist_coo.append([b, rx, hnode])
        hlist_train_coo.extend(hlist_coo)
    hlist_train_coo = list(zip(*hlist_train_coo))
    return hlist_train_coo

    
def check_feasibility(H, z):
    """
    Feasibility for hyperedge constraint
    """
    
    RHS_const = H.to_dense().squeeze(0).T.sum(dim=1) - 1
    LHS_const = H.to_dense().squeeze(0).T
    is_sat = (LHS_const @ torch.round(z).unsqueeze(-1)).squeeze(-1) <= RHS_const
    feasibility = (torch.sum(is_sat)/is_sat.numel()).item()
    return feasibility

def visualize_hyperedges(locs, hlist, buffer_distance, xy_lim=500):
    
    nNodes = locs[0].shape[0]
    transmitters, receivers = locs
    # Separate x and y coordinates
    x_coords = np.concatenate([transmitters[:, 0], receivers[:, 0]])
    y_coords = np.concatenate([transmitters[:, 1], receivers[:, 1]])

    # Indices that define certain polygons
    index_lists = hlist

    # Define the limits you want for x and y
    x_min, x_max = -xy_lim-10, xy_lim+10
    y_min, y_max = -xy_lim-10, xy_lim+10

    # ---------------------------------------------------------
    # 3) Create a figure
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))

    # ---------------------------------------------------------
    # 4) Scatter-plot all points (optional visualization)
    # ---------------------------------------------------------
    ax.scatter(x_coords[:nNodes], y_coords[:nNodes], color='blue', s=100, zorder=3, alpha=0.6)
    ax.scatter(x_coords[nNodes:], y_coords[nNodes:], color='red', s=100, zorder=3, alpha=0.6)

    # plot black line between each transmitter and receiver
    for i in range(nNodes):
        ax.plot([transmitters[i, 0], receivers[i, 0]], [transmitters[i, 1], receivers[i, 1]], color='black', zorder=1)
        
    # Label each point with its index
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        ax.text(x, y, str(i), color='white', ha='center', va='center', zorder=4)

    # ---------------------------------------------------------
    # 5) For each subset, build a rounded polygon using:
    #    MultiPoint -> convex_hull -> buffer(...)
    # ---------------------------------------------------------
    colors = ["red", "green", "orange", "purple", "cyan"]
    buffer_resolution = 10     # higher -> smoother circles around corners

    # plot a circle with radius xy_lim
    circle = plt.Circle((0, 0), xy_lim, color='black', fill=False, zorder=0)
    ax.add_artist(circle)

    for idx, node_list in enumerate(index_lists):
        # Extract the (x, y) coords for this subset
        subset_points = [(x_coords[i], y_coords[i]) for i in node_list]

        # Build a MultiPoint
        mp = MultiPoint(subset_points)
        
        # Compute the convex hull
        hull = mp.convex_hull  # Shapely polygon or multipolygon

        # Buffer the hull to get a rounded shape
        shape = hull.buffer(buffer_distance, resolution=buffer_resolution)

        # shape might be Polygon, MultiPolygon, or possibly something else
        if shape.geom_type == "Polygon":
            polygons = [shape]
        elif shape.geom_type == "MultiPolygon":
            polygons = list(shape)
        else:
            polygons = []

        # Choose a color from our list
        color = colors[idx % len(colors)]
        
        # Plot each polygon
        for poly in polygons:
            exterior_xy = list(poly.exterior.coords)
            patch = MplPolygon(
                exterior_xy,
                closed=True,
                facecolor='none',
                edgecolor=color,
                alpha=0.8,
                linewidth=1.5,
                zorder=2
            )
            ax.add_patch(patch)
        

    # ---------------------------------------------------------
    # 6) Final plot settings
    # ---------------------------------------------------------
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', 'box')
    ax.set_title("Hypergraph visualization")

    plt.show()
