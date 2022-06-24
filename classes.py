import torch.optim as optim
import torch.nn as nn
import torch


PATH = "./networks/recognizer.pth"


class Recognizer(nn.Module):
    def __init__(self):
        super(Recognizer, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(784, 40),
            nn.ReLU(),
            nn.Linear(40, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.Softmax(1),
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x) -> torch.Tensor:
        return self.sequential(x)


class OneOutRecognizer(nn.Module):
    def __init__(self):
        super(OneOutRecognizer, self).__init__()
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


r = Recognizer()
