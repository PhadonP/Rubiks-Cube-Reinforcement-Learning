import torch
import torch.nn as nn


class CubeNet(nn.Module):
    def __init__(self, cubeSize):
        super(CubeNet, self).__init__()
        channelsIn = cubeSize ** 2 * 36

        self.layers = nn.Sequential(
            nn.Linear(channelsIn, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 1)
        )

    def forward(self, states):
        return self.layers(states.float()).squeeze()
