import torch
import torch.nn as nn


class CubeNet(nn.Module):
    def __init__(self, cubeSize):
        super(CubeNet, self).__init__()
        channelsIn = cubeSize ** 2 * 36
        self.resBlocks = nn.ModuleList()

        self.firstBlock = nn.Sequential(
            nn.Linear(channelsIn, 5000),
            nn.BatchNorm1d(5000),
            nn.ReLU(),
            nn.Linear(5000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
        )

        if cubeSize == 3:
            numResBlocks = 4
        elif cubeSize == 2:
            numResBlocks = 1

        for _ in range(numResBlocks):
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
            nn.BatchNorm1d(channelsIn),
            nn.ReLU(),
            nn.Linear(channelsIn, channelsIn),
            nn.BatchNorm1d(channelsIn),
        )

        self.ReLU = nn.ReLU()

    def forward(self, states):
        return self.ReLU(self.layers(states) + states)
