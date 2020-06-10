import torch
from torch.utils.data import Dataset
import time
from search.BWAS import batchedWeightedAStarSearch
import pandas as pd
import numpy as np


def makeTrainingData(environment, preparedData, net, device):

    goalStates, exploredStates, validNextStates, goalNextStates = preparedData

    validExploredStates = exploredStates[validNextStates & ~goalNextStates]
    validExploredStatesOneHot = environment.oneHotEncoding(
        validExploredStates).to(device)

    MovesToGo = net(validExploredStatesOneHot)

    targets = torch.zeros(exploredStates.shape[:2])
    targets[validNextStates & ~goalNextStates] = MovesToGo.cpu()
    targets[goalNextStates] = 0
    targets[~validNextStates] = float("inf")

    targets = targets.min(1).values + 1

    targets[goalStates] = 0

    return targets.detach()


def prepareTrainingData(environment, numStates, scrambleDepth):

    states = environment.generateScrambles(numStates, scrambleDepth)

    goalStates = environment.checkIfSolved(states)

    exploredStates, validNextStates, goalNextStates = environment.exploreNextStates(
        states
    )

    encodedStates = environment.oneHotEncoding(states)

    return encodedStates.detach(), [goalStates, exploredStates, validNextStates, goalNextStates]


class Puzzle15DataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train(net, device, loader, optimizer):

    net.train()
    lossLogger = []
    valueLogger = []

    for i, (x, y) in enumerate(loader):

        x = x.to(device)
        y = y.to(device)
        fx = net(x)

        loss = torch.nn.MSELoss()(fx, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lossLogger.append(loss.item())

        avgValue = torch.mean(fx).item()
        valueLogger.append(avgValue)

        print(
            "TRAINING: | Iteration [%d/%d] | Loss %.2f | Average Value %.3f"
            % (i + 1, len(loader), loss.item(), avgValue)
        )

    meanLoss = sum(lossLogger) / len(lossLogger)
    meanValue = sum(valueLogger) / len(valueLogger)
    return meanLoss, meanValue


def test(epoch, env, net, device, config, filePath, verbose):

    torch.set_num_threads(1)

    net.to(device)
    net.eval()

    numScrambles = config.numTestScrambles
    maxScrambleDepth = config.testScrambleDepth

    columns = ["Epoch", "Scramble Depth", "Time", "isSolved",
               "Move Count", "Nodes Generated", "Search Iterations"]
    df = pd.DataFrame(columns=columns)

    for i in range(numScrambles):

        scrambleDepth = ((i + 1) * (maxScrambleDepth - 2)) // numScrambles + 2
        scramble = env.generateScramble(scrambleDepth)

        (
            moves,
            numNodesGenerated,
            searchItr,
            isSolved,
            solveTime,
        ) = batchedWeightedAStarSearch(
            scramble,
            config.depthWeight,
            config.numParallel,
            env,
            net,
            device,
            config.maxSearchItr,
            verbose
        )

        if isSolved:
            row = [epoch, scrambleDepth, solveTime, isSolved,
                   len(moves), numNodesGenerated, searchItr]
        else:
            row = [epoch, scrambleDepth, solveTime, isSolved,
                   np.nan, numNodesGenerated, searchItr]

        df.loc[len(df)] = row

    with open(filePath, 'a') as f:
        df.to_csv(f, header=f.tell() == 0, index=False)
