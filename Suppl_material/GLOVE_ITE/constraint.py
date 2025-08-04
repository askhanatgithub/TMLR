# Code implementaion is taken from Dynamic narrowing of vae bottlenecks using geco and l0 regularization. 2021 International Joint Conference on Neural Networks (IJCNN),pages 1–8, 2020.
from torch.autograd import Variable
from torch.nn import functional as F
import torch
from torch import nn
from torch.autograd import Function
class Constraint(nn.Module):
    def __init__(self, tolerance, device, lambda_min=0.0, lambda_max=1.0, lambda_init=1.0, alpha=0.99):
        super(Constraint, self).__init__()
        self.moving_average = None
        self.tolerance = tolerance
        self.device = device
        self.lambd_min = nn.Parameter(torch.FloatTensor([lambda_min]), requires_grad=False).to(device)
        self.lambd_max = nn.Parameter(torch.FloatTensor([lambda_max]), requires_grad=False).to(device)
        self.alpha = alpha
        self.clamp = ClampFunction()
        self.floatTensor = torch.FloatTensor if self.device == 'cpu' else torch.cuda.FloatTensor

        self.lambd = nn.Parameter(self._inv_squared_softplus(self.floatTensor([lambda_init])), requires_grad=True).to(device)
        self.multiplier = F.softplus(self.lambd)**2

        self.constraint_is_hit = False
        self.constraint_has_been_hit = False

    def forward(self, value):
        constraint = value - self.tolerance
        self.constraint_is_hit = constraint < 0
        self.constraint_has_been_hit = self.constraint_has_been_hit or (constraint < 0)

        with torch.no_grad():
            if self.moving_average is None:
                self.moving_average = constraint
            else:
                self.moving_average = self.alpha * self.moving_average + (1 - self.alpha) * constraint

        cost = constraint + (self.moving_average - constraint).detach()

        # we use squared softplus as in
        # https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/optimization_constraints.py
        # we also clamp the resulting values
        self.multiplier = self.clamp.apply(F.softplus(self.lambd)**2, self.lambd_min, self.lambd_max).to(self.device)
        
        return self.multiplier * cost, self.multiplier.item()

    def get_multiplier(self):
        return self.multiplier.item()

    def _inv_squared_softplus(self, x):
        sqrt = torch.sqrt(x)
        return torch.log(torch.exp(sqrt) - 1.0)


class ClampFunction(Function):
    '''
    Clamp a value between min and max.
    When the gradients push the value further away from the [min,max] range, set to zero
    When the gradients push the value back in the [min,max] range, let them flow through
    '''

    @staticmethod
    def forward(ctx, lambd, min_value, max_value):
        ctx.save_for_backward(lambd, min_value, max_value)
        if lambd < min_value:
            return min_value
        elif lambd > max_value:
            return max_value
        else:
            return lambd

    @staticmethod
    def backward(ctx, lambd_grad):
        lambd, min_value, max_value = ctx.saved_tensors
 
        if lambd < min_value and lambd_grad < 0.0:
            lambd_grad[:] = 0.0
        elif lambd > max_value and lambd_grad > 0.0:
            lambd_grad[:] = 0.0
        else:
            lambd_grad = -lambd_grad

        return lambd_grad, None, None

class lamd(nn.Module):
    def __init__(self):
        super(lamd, self).__init__()
        self.lmd = nn.Parameter(torch.tensor(1.0))  # Register parameter

    def forward(self, x):
        return x * self.lmd
        
    def get_lambd(self):
        return self.lmd.item()