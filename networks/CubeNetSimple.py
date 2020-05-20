import torch
import torch.nn as nn


class CubeNet(nn.Module):
    def __init__(self, cubeSize):
        super(CubeNet, self).__init__()
        channelsIn = cubeSize ** 2 * 36

        self.layers = nn.Sequential(
            nn.Linear(channelsIn, 5000),
            nn.ReLU(),
            nn.BatchNorm1d(5000),
            nn.Linear(5000, 3000),
            nn.ReLU(),
            nn.BatchNorm1d(3000),
            nn.Linear(3000, 3000),
            nn.ReLU(),
            nn.BatchNorm1d(3000),
            nn.Linear(3000, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 1),
        )

    def forward(self, states):
        return self.layers(states.float()).squeeze()
