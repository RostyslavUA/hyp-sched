from torch.utils.data import Dataset, DataLoader
from utils import get_data, get_H
import torch 
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


def hypergraph_generation(V_H, I, hyperedges):
    hypergraph = {}

    

    hypergraph["I"] = torch.DoubleTensor(I)


    H = torch.DoubleTensor(hyperedges)
    Dv_inv = torch.diag(H.sum(0)**(-1/2))
    De_inv = torch.diag(H.sum(1)**(-1))
    W = torch.eye(V_H, dtype=torch.float64)
    hypergraph["H"] = H
    hypergraph["Dv_inv"] = Dv_inv
    hypergraph["De_inv"] = De_inv
    hypergraph["W"] = W
    return hypergraph

def get_hyper_dataset(V_H, N, xy_lim, theta, k, Samples):
    itens, hlist = get_data(Samples, V_H, N, xy_lim, theta, k)
    H = get_H(hlist, V_H, Samples)


    Is = torch.zeros((Samples, V_H, V_H), dtype=torch.float64)
    Dv_invs = torch.zeros((Samples, V_H, V_H), dtype=torch.float64)
    De_invs = torch.zeros((Samples, V_H, V_H), dtype=torch.float64)
    Hs = torch.zeros((Samples, V_H, V_H), dtype=torch.float64)
    Ws = torch.zeros((Samples, V_H, V_H), dtype=torch.float64)

    i = 0
    for I, hyperedges in zip(itens, H):
        hyp = hypergraph_generation(V_H, I, hyperedges)
        Is[i] = hyp["I"]
        Dv_invs[i] = hyp["Dv_inv"]
        De_invs[i] = hyp["De_inv"]
        Hs[i] = hyp["H"]
        Ws[i] = hyp["W"]
        i += 1

    dataset_samples = HyperDataset(Is, Dv_invs, De_invs, Hs, Ws)
    return dataset_samples

