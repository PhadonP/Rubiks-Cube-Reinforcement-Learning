import torch
from torch.utils.data import Dataset


def makeTrainingData(environment, net, device, numStates, scrambleDepth):

    states, _ = environment.generateScrambles(numStates, scrambleDepth)
    goalStates = torch.all(states == environment.solvedState, 2)
    goalStates = goalStates.all(1)

    exploredStates, validNextStates, goalNextStates = environment.exploreNextStates(
        states
    )

    validExploredStates = exploredStates[validNextStates & ~goalNextStates]
    validExploredStatesOneHot = environment.oneHotEncoding(validExploredStates).to(
        device
    )

    MovesToGo = net(validExploredStatesOneHot)

    targets = torch.zeros(exploredStates.shape[:2])
    targets[validNextStates & ~goalNextStates] = MovesToGo
    targets[goalNextStates] = 0
    targets[~validNextStates] = float("inf")

    targets = targets.min(1).values + 1

    targets[goalStates] = 0

    encodedStates = environment.oneHotEncoding(states)

    return encodedStates.detach(), targets.detach()


class Puzzle15DataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train(net, device, loader, optimizer, lossLogger):

    net.train()
    epochLoss = 0

    for i, (x, y) in enumerate(loader):

        x = x.to(device)
        y = y.to(device)
        fx = net(x)

        loss = torch.nn.MSELoss()(fx, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epochLoss += loss.item()

        lossLogger.append(loss.item())

        print(
            "TRAINING: | Iteration [%d/%d] | Loss %.2f |"
            % (i + 1, len(loader), loss.item())
        )

    # return the average loss from the epoch as well as the logger array
    return epochLoss / len(loader), lossLogger
