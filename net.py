import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, channelsIn):
        super(Net, self).__init__()

        self.convlayers = nn.Sequential(
            nn.Conv2d(channelsIn, 100, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(100, 200, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(200, 100, kernel_size=2, stride=1, padding=0),
        )

        self.linear = nn.Sequential(
                        nn.Linear(900, 100),
                        nn.ReLU(),
                        nn.Linear(100, 1)
        )

    def forward(self, input):
        outconv = self.convlayers(input)

        flattened = outconv.view(outconv.shape[0], -1)
        return self.linear(flattened)
        

