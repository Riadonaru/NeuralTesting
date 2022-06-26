import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


PATH = "./networks/recognizer.pth"


class Recognizer(nn.Module):
    def __init__(self):
        super(Recognizer, self).__init__()
        # self.sequential = nn.Sequential(
        #     nn.Linear(784, 40),
        #     nn.ReLU(),
        #     nn.Linear(40, 30),
        #     nn.ReLU(),
        #     nn.Linear(30, 20),
        #     nn.ReLU(),
        #     nn.Linear(20, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 10),
        #     nn.Softmax(1),
        # )
        self.conv1 = nn.Conv1d(1, 3, 5)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(3, 6, 5)
        self.fc1 = nn.Linear(193 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(1, 6 * 193)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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
