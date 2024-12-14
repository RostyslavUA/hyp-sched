import torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN, ReLU6 as ReLU6
from torch.autograd import Variable

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