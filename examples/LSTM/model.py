import random
import os
import torch


class LSTM(torch.nn.Module):
    def __init__(self, inputnum, hnum):
        super().__init__()
        self.hnum = hnum
        self.wx = torch.randn((inputnum, 4 * hnum), requires_grad=True) / inputnum ** (1 / 2)
        self.wh = torch.randn((hnum, 4 * hnum), requires_grad=True) / hnum ** (1 / 2)
        self.b = torch.rand([1, 4 * hnum], requires_grad=True)

        self.wx = torch.nn.Parameter(self.wx)
        self.wh = torch.nn.Parameter(self.wh)
        self.b = torch.nn.Parameter(self.b)

    def forward(self, fullx, h, c):  # x =[batch,L,in]
        self.fullh = []
        self.h = h
        self.c = c
        for i in range(fullx.shape[1]):
            self.x = fullx[:, i, :]
            self.t = torch.mm(self.x, self.wx) + torch.mm(self.h, self.wh) + self.b  # t=[batch,4h])

            self.gate = torch.sigmoid(self.t[:, :3 * self.hnum])  # gate=[batch,3h])
            f, i, o = [self.gate[:, n * self.hnum:(n + 1) * self.hnum] for n in range(3)]
            g = torch.tanh(self.t[:, 3 * self.hnum:4 * self.hnum])
            self.c = self.c * f
            self.c = self.c + g * i
            self.h = torch.tanh(self.c) * o

            self.fullh.append(self.h.unsqueeze(1))
        self.fullh = torch.cat(self.fullh, dim=1)
        return self.fullh, self.h, self.c  # fullh=[batch,L,h] #h=[batch,]

    def __call__(self, x, h, c):
        return self.forward(x, h, c)


if __name__ == "__main__":
    rnn = LSTM(10, 3)

    y, h, c = rnn(torch.ones([1, 100, 10]), torch.ones([1, 3]), torch.ones([1, 3]))
    print(h)
    a = torch.zeros([1, 100, 3])
    a[0, -1, :] = 1
    print(a)
    y.backward(a)
    print(rnn.wh.grad)
