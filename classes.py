import torch.optim as optim
import torch.nn as nn
import torch
import data
import numpy as np


PATH = "./networks/recognizer.pth"


class Recognizer(nn.Module):
    def __init__(self):
        super(Recognizer, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(784, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x) -> torch.Tensor:
        return self.sequential(x)


class OneOutRecognizer(nn.Module):
    def __init__(self):
        super(Recognizer, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(784, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.ReLU(),
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x) -> torch.Tensor:
        return self.sequential(x)


r = OneOutRecognizer()
