from torch import nn
import torch.nn.functional as F
import torch
from torch.autograd import Function


class OneLevelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, th):
        out = (input >= 1. * th).float() * th
        input = ((input.detach() >= 0.5 * th) * (input.detach() <= 1.5 * th)).float()       
        ctx.save_for_backward(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (tmp,) = ctx.saved_tensors
        grad_input = grad_output * tmp
        return grad_input, None


class TwoLevelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, th):
        out2 = (input >= 2. * th).float()
        out1 = (input >= 1. * th).float() * (1. - out2)
        out = out1 * th + out2 * 2. * th
        input = ((input.detach() >= 0.5 * th) * (input.detach() <= 2.5 * th)).float()       
        ctx.save_for_backward(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (tmp,) = ctx.saved_tensors
        grad_input = grad_output * tmp
        return grad_input, None


class FourLevelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, th):       
        out4 = (input >= 4. * th).float()
        out3 = (input >= 3. * th).float() * (1. - out4)
        out2 = (input >= 2. * th).float() * (1. - out4) * (1. - out3)
        out1 = (input >= 1. * th).float() * (1. - out4) * (1. - out3) * (1. - out2)
        out = out1 * th + out2 * 2. * th + out3 * 3. * th + out4 * 4. * th
        input = ((input.detach() >= 0.5 * th) * (input.detach() <= 4.5 * th)).float()
        ctx.save_for_backward(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (tmp,) = ctx.saved_tensors
        grad_input = grad_output * tmp
        return grad_input, None
        


class LMHTNeuron(nn.Module):
    def __init__(self, L: int, T=2, th=1., inital_mem=0.):
        super(LMHTNeuron, self).__init__()
        self.v_threshold = nn.Parameter(torch.tensor([th]), requires_grad=False)
        self.v = None
        self.inital_mem = inital_mem * th

        if L == 2:
            self.act = TwoLevelFunction.apply
        elif L == 4:
            self.act = FourLevelFunction.apply

        self.alpha = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        self.mask = nn.Parameter(torch.zeros((T, T, 1, 1, 1, 1)), requires_grad=True)
        #self.mask_linear = nn.Parameter(torch.zeros((T, T, 1, 1)), requires_grad=True)
        self.mask_linear = nn.Parameter(torch.zeros((T, T, 1, 1, 1)), requires_grad=True)
        self.T = T
        self.scale = 1.
        
        
    def forward(self, x):
        self.v = torch.ones_like(x[0]) * self.inital_mem
        x = x * self.scale
        
        if len(x.shape) == 5:
            self.core = self.mask
        else:
            self.core = self.mask_linear

        spike_pot = []
        for t in range(self.T):
            self.v = (self.alpha.sigmoid() + 0.5) * self.v.detach() + ((2 * self.core[t].sigmoid() / x.shape[0]) * x).sum(dim=0)
            output = self.act(self.v, self.v_threshold)
            self.v -= output.detach()
            spike_pot.append(output)

        return torch.stack(spike_pot, dim=0)



class LMHT_Inference_Neuron(nn.Module):
    def __init__(self, L, alpha, mask, mask_linear, T=4, th=1., inital_mem=0.):
        super(LMHT_Inference_Neuron, self).__init__()
        self.v_threshold = nn.Parameter(torch.tensor([th]), requires_grad=False)
        self.v = None
        self.inital_mem = inital_mem * th

        self.alpha = nn.Parameter(torch.ones(T), requires_grad=False)
        self.mask = nn.Parameter(torch.zeros((T, T, 1, 1, 1, 1)), requires_grad=False)
        self.mask_linear = nn.Parameter(torch.zeros((T, T, 1, 1)), requires_grad=False)
        self.T = T
        self.scale = 1.
        
        self.reparameterization(L, alpha, mask, mask_linear)
        
    def reparameterization(self, L, alpha, mask, mask_linear):
        for t in range(0,self.T,L):
            self.alpha[t] = alpha.sigmoid().item() + 0.5
            for j in range(0,self.T,L):
                self.mask[t:t+L,j:j+L] = 2.*mask[t//L,j//L].sigmoid().item() / self.T
                self.mask_linear[t:t+L,j:j+L] = 2.*mask_linear[t//L,j//L].sigmoid().item() / self.T
        
        
    def forward(self, x):
        self.v = torch.ones_like(x[0]) * self.inital_mem
        x = x * self.scale
        
        if len(x.shape) == 5:
            self.core = self.mask
        else:
            self.core = self.mask_linear

        spike_pot = []
        for t in range(self.T):
            self.v = self.alpha[t] * self.v + (self.core[t] * x).sum(dim=0)
            output = (self.v >= self.v_threshold) * self.v_threshold
            self.v -= output
            spike_pot.append(output)

        return torch.stack(spike_pot, dim=0)


class FloorLayer(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    

class IFNeuron(nn.Module):
    def __init__(self, T=2, th=1., inital_mem=0.):
        super(IFNeuron, self).__init__()
        self.v_threshold = nn.Parameter(torch.tensor([th]), requires_grad=False)
        self.v = None
        self.inital_mem = inital_mem * th
        self.T = T
        self.act = OneLevelFunction.apply
        
        #self.act = TwoLevelFunction.apply
        
    def forward(self, x):
        self.v = torch.ones_like(x[0]) * self.inital_mem        
        spike_pot = []
        for t in range(self.T):
            self.v = self.v + x[t]
            output = self.act(self.v, self.v_threshold)
            self.v -= output
            spike_pot.append(output)

        return torch.stack(spike_pot, dim=0)



qcfs = FloorLayer.apply


class QCFS(nn.Module):
    def __init__(self, up=1., t=4):
        super().__init__()
        #self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.thresh = nn.Parameter(torch.tensor([up]), requires_grad=True)
        
        self.t = t
    def forward(self, x):
        #x = x / self.up
        x = x / self.thresh
        
        x = qcfs(x*self.t+0.5)/self.t
        x = torch.clamp(x, 0, 1)
        #x = x * self.up
        x = x * self.thresh
        return x
    
