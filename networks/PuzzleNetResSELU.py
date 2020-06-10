import torch
import torch.nn as nn


class PuzzleNet(nn.Module):
    def __init__(self, puzzleSize):
        super(PuzzleNet, self).__init__()

        channelsIn = (puzzleSize + 1) ** 2
        self.resBlocks = nn.ModuleList()

        self.firstBlock = nn.Sequential(
            nn.Linear(channelsIn, 1000),
            nn.SELU(),
            nn.BatchNorm1d(1000),
        )

        for _ in range(3):
            self.resBlocks.append(ResidualBlock(1000))

        self.finalLayer = nn.Linear(1000, 1)

    def forward(self, states):

        out = self.firstBlock(states.float())

        for block in self.resBlocks:
            out = block(out)

        out = self.finalLayer(out)

        return out.squeeze()


class ResidualBlock(nn.Module):
    def __init__(self, channelsIn):
        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(channelsIn, channelsIn),
            nn.SELU(),
            nn.BatchNorm1d(channelsIn),
            nn.Linear(channelsIn, channelsIn),
            nn.BatchNorm1d(channelsIn),
        )

        self.SELU = nn.SELU()

    def forward(self, states):
        return self.SELU(self.layers(states) + states)
