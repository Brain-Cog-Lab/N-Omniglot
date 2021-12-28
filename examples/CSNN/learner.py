import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.5  # neuronal threshold
lens = 0.5    # hyper-parameters of approximate function
decay = 0.2   # decay constants


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """
    
    scale = 100.0 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad
    
# here we overwrite our naive spike function by the "SurrGradSpike" nonlinearity which implements a surrogate gradient
spike_fn  = SurrGradSpike.apply

# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply
# membrane potential update

# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


class LIFConv(nn.Module):
    def __init__(self,in_planes, out_planes, kernel_size, stride, padding, decay=0.2):
        super( ).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.mem = self.spike = None
        self.decay=decay

    def mem_update(self, x):
        if self.mem is None:
            self.mem = torch.zeros_like(x, device=device)
            self.spike = torch.zeros_like(x, device=device)

        self.mem = self.mem * self.decay * (1. - self.spike) + x
        self.spike = act_fun(self.mem)  # act_fun : approximation firing function
        return self.spike
        
    def forward(self, x ):
        x = self.conv(x)
        x = self.mem_update(x)
        return x
        
    def reset(self):
        self.mem = self.spike = None


class LIFLinear(nn.Module):
    def __init__(self, in_planes, out_planes, decay=0.2, last_layer=False):
        super().__init__()
        self.fc = nn.Linear(in_planes, out_planes)
        self.mem = self.spike = None
        self.decay = decay
        self.last_layer = last_layer

    def mem_update(self, x):
        if self.mem is None:
            self.mem = torch.zeros_like(x, device=device)
            self.spike = torch.zeros_like(x, device=device)
        if self.last_layer:
            self.mem = self.mem + x
        else:
            self.mem = self.mem * self.decay * (1. - self.spike) + x
            self.spike = act_fun(self.mem)  # act_fun : approximation firing function
        return self.spike

    def forward(self, x):
        x = self.fc(x)
        x = self.mem_update(x)
        return x

    def reset(self):
        self.mem = self.spike = None


class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        self.conv1 = LIFConv(2, 15, kernel_size=5, stride=1, padding=0)
        self.conv2 = LIFConv(15, 40, kernel_size=5, stride=1, padding=0)
         
        self.fc1 = LIFLinear(640, 300)
        self.fc2 = LIFLinear(300, 1623,0.2,True)

    def forward(self, input ):
        for step in range(input.shape[1]):  # simulation time steps
            x = input[:,step]
            x = self.conv1(x.float())
            x = F.avg_pool2d(x, 2)
            x = self.conv2(x)
            x = F.avg_pool2d(x, 2) 

            x = x.view(input.shape[0], -1)
            x = self.fc1(x)
            x = self.fc2(x)

        outputs = self.fc2.mem / input.shape[1]
        self.reset()
        return outputs

    def reset(self):
        for i in self.children():
            i.reset()

