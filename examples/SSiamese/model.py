import torch
import torch.nn as nn
import torch.nn.functional as F


# define approximate firing function
thresh, lens = 0.5, 0.5
decay = 0.2


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
def mem_update(ops, x, mem, spike):
    mem = mem * decay * (1. - spike)  # + ops(x)
    mem = mem + ops(x)
    spike = act_fun(mem)  # act_fun : approximation firing function
    return mem, spike


class SSiamese(nn.Module):
    def __init__(self, device):
        super(SSiamese, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)

        self.liner = nn.Linear(56448, 4096)
        self.out = nn.Linear(4096, 1)

    def forward(self, input1, input2, batch_size=128, time_window=10):
        device = self.device
        c1_mem1 = c1_spike1 = torch.zeros(batch_size, 64, 96, 96, device=device)
        c2_mem1 = c2_spike1 = torch.zeros(batch_size, 128, 42, 42, device=device)

        c1_mem2 = c1_spike2 = torch.zeros(batch_size, 64, 96, 96, device=device)
        c2_mem2 = c2_spike2 = torch.zeros(batch_size, 128, 42, 42, device=device)
        h1_mem2 = h1_spike2  = torch.zeros(batch_size, 4096, device=device)

        outputs = torch.zeros(batch_size, 1, device=device)

        for step in range(time_window):  # simulation time steps
            x = input1[step]
            c1_mem1, c1_spike1 = mem_update(self.conv1, x, c1_mem1, c1_spike1)
            x = F.avg_pool2d(c1_spike1, 2)
            c2_mem1, c2_spike1 = mem_update(self.conv2, x, c2_mem1, c2_spike1)
            x = F.avg_pool2d(c2_spike1, 2)
            x1 = x.view(batch_size, -1)

            x = input2[step]
            c1_mem2, c1_spike2 = mem_update(self.conv1, x, c1_mem2, c1_spike2)
            x = F.avg_pool2d(c1_spike2, 2)
            c2_mem2, c2_spike2 = mem_update(self.conv2, x, c2_mem2, c2_spike2)
            x = F.avg_pool2d(c2_spike2, 2)
            x = x.view(batch_size, -1)

            h1_mem2, h1_spike2 = mem_update(self.liner, torch.abs(x-x1), h1_mem2, h1_spike2)
            outputs += self.out(h1_spike2)

        return outputs


# for test
if __name__ == '__main__':
    net = SSiamese()
    print(net)
    print(list(net.parameters()))
