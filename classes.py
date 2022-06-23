import torch.optim as optim
import torch.nn as nn
import torch
import data
import numpy as np


PATH = "./recognizer.pth"


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

    def forward(self, x) -> torch.Tensor:
        return self.sequential(x)


r = Recognizer()

criterion = nn.MSELoss()
optimizer = optim.SGD(r.parameters(), lr=0.001, momentum=0.9)

if __name__ == "__main__":
    try:
        for epoch in range(3):
            running_loss = 0.0
            for i, tup in enumerate(data.get_next()):
                input, target = tup
                # zero the parameter gradients

                optimizer.zero_grad()

                # forward + backward + optimize
                output = r.forward(input)
                # print("Out:\n", output, "\nTarget:\n", target, "\n~~~\n")
                # if i == 3:
                #     break
                # print("Target:\n", target, "\nOutput:\n", output)
                loss: torch.Tensor = criterion(output, target)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0.0
    finally:
        torch.save(r.state_dict(), PATH)
        print("Saved data at: %s" % PATH)
