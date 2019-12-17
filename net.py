import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, inputShape):
        super(Net, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(inputShape, 4096),
            nn.ELU(),
            nn.Linear(4096, 2048),
            nn.ELU(),
            nn.Linear(2048, 512),
            nn.ELU(),
            nn.Linear(512, 1)
        )

        self.test = nn.Sequential(
            nn.Linear(inputShape, 1)
        )

    def forward(self, input):
        return self.test(input)

