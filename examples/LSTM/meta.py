from torch import optim
from examples.LSTM.learner import *


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        super(Meta, self).__init__()

        self.net = net()
        self.optim = optim.SGD(self.net.parameters(), lr=0.1)

        self.onehot = torch.eye(1623).cuda()
        self.lossfunction = nn.CrossEntropyLoss()

    def forward(self, x, y):
        self.net.train()
        logits = self.net(x)
        loss = self.lossfunction(logits, y)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        pred_q = F.softmax(logits, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y).sum().item() / pred_q.shape[0]  # convert to numpy

        return correct

    def test(self, x, y):
        self.net.eval()

        logits = self.net(x)

        pred_q = F.softmax(logits, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y).sum().item() / pred_q.shape[0]  # convert to numpy

        return correct
