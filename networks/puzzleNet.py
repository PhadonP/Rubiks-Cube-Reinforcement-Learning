import torch
import torch.nn as nn


class PuzzleNet(nn.Module):
    def __init__(self, channelsIn):
        super(PuzzleNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear((channelsIn + 1) ** 2, 2000),
            nn.ReLU(),
            nn.BatchNorm1d(2000),
            nn.Linear(2000, 1500),
            nn.ReLU(),
            nn.BatchNorm1d(1500),
            nn.Linear(1500, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 1),
        )

    def forward(self, states):
        return self.layers(states).squeeze()

    #     self.convLayers = nn.Sequential(
    #         nn.Conv2d(channelsIn + 1, 100, kernel_size=2, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.BatchNorm2d(100),
    #         nn.Conv2d(100, 200, kernel_size=2, stride=1, padding=0),
    #         nn.ReLU(),
    #         nn.BatchNorm2d(200),
    #         nn.Conv2d(200, 100, kernel_size=2, stride=1, padding=0),
    #         nn.ReLU(),
    #         nn.BatchNorm2d(100),
    #     )

    #     self.linear = nn.Sequential(
    #         nn.Linear(self.getConvOutput(channelsIn + 1), 100),
    #         nn.ReLU(),
    #         nn.BatchNorm1d(100),
    #         nn.Linear(100, 1),
    #     )

    # def getConvOutput(self, channelsIn):
    #     sample = torch.zeros(
    #         1, channelsIn, int(channelsIn ** 0.5), int(channelsIn ** 0.5)
    #     )

    #     outConv = self.convLayers(sample)
    #     return outConv.view(outConv.shape[0], -1).shape[1]

    # def forward(self, states):
    #     outConv = self.convLayers(states)
    #     flattened = outConv.view(outConv.shape[0], -1)
    #     return self.linear(flattened).squeeze()
