import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np
from examples.SMAML.learner import SCNN2
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = SCNN2(device)

        self.optimmeta = optim.SGD(self.net.parameters(), lr=self.meta_lr)

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """

        criterion = nn.CrossEntropyLoss()
        grad_q = None

        corrects = [0 for _ in range(self.update_step)]
        lossfull = None
        for i in range(x_spt.shape[0]):
            logits = self.net(x_spt[i])
            loss = criterion(logits, y_spt[i])
            self.optimmeta.zero_grad()
            loss.backward()
            nettemp = deepcopy(self.net)
            paratemp = {i: torch.clone(j) for i, j in self.net.named_parameters()}
            nettemp.load_state_dict(paratemp, strict=False)

            optimtemp = optim.SGD(nettemp.parameters(), lr=self.update_lr)
            for g, p in zip(nettemp.parameters(), self.net.parameters()):
                g.grad = p.grad
            optimtemp.step()  # 更新了fast weight
            for j in range(self.update_step):
                logits = nettemp(x_spt[i])
                loss = criterion(logits, y_spt[i])
                optimtemp.zero_grad()
                loss.backward()
                optimtemp.step()

            logits = nettemp(x_qry[i])
            loss = criterion(logits, y_qry[i])
            optimtemp.zero_grad()
            if lossfull is None:
                lossfull = loss / self.task_num
            else:
                lossfull += loss / self.task_num

            with torch.no_grad():
                pred_q = F.softmax(logits, dim=1).argmax(dim=1)
                # _, q = y_qry[i] .max(1)
                q = y_qry[i]
                correct = torch.eq(pred_q, q).sum().item()  # convert to numpy
                corrects[0] = corrects[0] + correct

        self.optimmeta.zero_grad()
        lossfull.backward()
        self.optimmeta.step()

        accs = np.array(corrects) / (self.k_qry * self.n_way * x_spt.shape[0]) 

        return accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        querysz = x_qry.size(0)
        criterion = nn.CrossEntropyLoss()
        corrects = [0 for _ in range(self.update_step_test)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        nettemp = deepcopy(self.net)

        optimtemp = optim.SGD(nettemp.parameters(), lr=self.update_lr)
        # 1. run the i-th task and compute loss for k=0
        logits = nettemp(x_spt)
        loss = criterion(logits, y_spt)
        optimtemp.zero_grad()
        loss.backward()
        optimtemp.step()
        # this is the loss and accuracy before first update

        for j in range(self.update_step_test):
            logits = nettemp(x_spt)
            loss = criterion(logits, y_spt)
            optimtemp.zero_grad()
            loss.backward()
            optimtemp.step()

            # [setsz, nway]
            logits_q = nettemp(x_qry)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[j] = corrects[j] + correct
        del nettemp

        accs = np.array(corrects) / querysz

        return accs
