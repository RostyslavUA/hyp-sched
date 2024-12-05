import torch, numpy as np
import torch.nn as nn, torch.nn.functional as F

from torch.autograd import Variable
import utils



class HyperGCN(nn.Module):
    def __init__(self, V, E, X, d, c, fast, mediators):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of links
        """
        super(HyperGCN, self).__init__()
        d, l, c = d, 3, c
        cuda = torch.cuda.is_available()

        h = [d]
        for i in range(l-1):
            h.append(2**l)
        h.append(c)

        if fast:
            reapproximate = False
            structure = utils.Laplacian(V, E, X, mediators)        
        else:
            reapproximate = True
            structure = E
            
        self.layers = nn.ModuleList([utils.HyperGraphConvolution(h[i], h[i+1], reapproximate, cuda) for i in range(l)])
        self.fc = torch.nn.Linear(c, 1)
        self.do, self.l = 0.1, 3
        self.structure, self.m = structure, mediators
        



    def forward(self, X):
        """
        an l-layer GCN
        """
        do, l, m = self.do, self.l, self.m
        
        for i, hidden in enumerate(self.layers):
            X = F.relu(hidden(self.structure, X, m))
            if i < l - 1:
                X = F.dropout(X, do, training=self.training)
        
        X = self.fc(X).squeeze(-1)
        return F.sigmoid(X) #F.log_softmax(X, dim=1)
