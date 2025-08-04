# Implementation of Learning sparse neural networks
#through l 0 regularization. In International Conference on
#Learning Representations, 2018.
# Code details are inspired by the implementation at https://github.com/moskomule/l0.pytorch
import torch
import math
from torch.autograd import Variable
from torch.nn import functional as F
from torch import nn
def hard_sigmoid(x):
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))

# Base class _L0Norm with L0 regularization
class _L0Norm(nn.Module):
    def __init__(self, origin, loc_mean=0, loc_sdev=0.01, beta=2 / 3, gamma=-0.1, zeta=1.1, fix_temp=True):
        super(_L0Norm, self).__init__()
        self._origin = origin
        self._size = self._origin.weight.size()
        self.loc = nn.Parameter(torch.zeros(self._size).normal_(loc_mean, loc_sdev))
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
        self.register_buffer("uniform", torch.zeros(self._size))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = math.log(-gamma / zeta)

    def _get_mask(self):
        
        self.uniform.uniform_()
        u = Variable(self.uniform)
        s = F.sigmoid((torch.log(u) - torch.log(1 - u) + self.loc) / self.temp)
        s = s * (self.zeta - self.gamma) + self.gamma
        penalty = F.sigmoid(self.loc - self.temp * self.gamma_zeta_ratio).sum()
        
        return hard_sigmoid(s), penalty

# L0Linear layer class
class L0Linear(_L0Norm):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(L0Linear, self).__init__(nn.Linear(in_features, out_features, bias=bias), **kwargs)

    def forward(self, input):
        mask, penalty = self._get_mask()
        return input*mask, penalty, mask

# Define a simple network with an L0Linear layer
class SparseNet(nn.Module):
    def __init__(self,inf,outf):
        super(SparseNet,self).__init__()
        self.fc1 = L0Linear(inf,outf)
       
    def forward(self, x):
        out, penalty1,mask = self.fc1(x)
        
        return out, penalty1,mask.detach()