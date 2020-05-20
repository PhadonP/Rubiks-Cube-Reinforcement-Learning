import torch
import torch.nn as nn


class PuzzleNet(nn.Module):
    def __init__(self, puzzleSize):
        super(PuzzleNet, self).__init__()

        channelsIn = (puzzleSize + 1) ** 2
        self.layers = nn.Sequential(
            nn.Linear(channelsIn, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 1)
        )

    def forward(self, states):
        return self.layers(states.float()).squeeze()
