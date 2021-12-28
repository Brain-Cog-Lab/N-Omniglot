import torch
import torch.nn as nn
import torch.nn.functional as F
from examples.LSTM.model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True)
        self.bn2 = torch.nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True)
        self.relu1 = torch.nn.ReLU(True)
        self.relu2 = torch.nn.ReLU(True)
        self.maxpool1 = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.maxpool2 = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.LSTM = LSTM(1568, 128)
        self.fc = torch.nn.Linear(128, 1623)

    def forward(self, input):
        h = torch.zeros([input.shape[0], 128], device=device)
        c = torch.zeros([input.shape[0], 128], device=device)

        result = None
        for step in range(input.shape[1]):  # simulation time steps

            result = input[:, step]
            result = self.conv1(result.float())
            result = self.bn1(result)
            result = self.relu1(result)
            result = self.maxpool1(result)

            result = self.conv2(result)  # +result
            result = self.bn2(result)
            result = self.relu2(result)
            result = self.maxpool2(result)
            result = result.reshape(input.shape[0], 1, -1)
            result, h, c = self.LSTM(result, h, c)
            result = result.reshape(input.shape[0], -1)

        result = self.fc(result)

        return result

    def __call__(self, x):
        return self.forward(x)
