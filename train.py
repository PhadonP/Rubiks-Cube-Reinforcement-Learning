from puzzleNEnv import PuzzleN
from net import Net
import torch
import torch.nn.functional as f

def makeTrainingData(environment, net, device, numStates, maxScramble):

    states = environment.generateScrambles(numStates, maxScramble)
    goalStates = torch.all(states == environment.solvedState, 2)
    goalStates = goalStates.all(1).to(device)

    exploredStates, validNextStates, goalNextStates = environment.exploreNextStates(states)

    validExploredStates = exploredStates[validNextStates & ~goalNextStates]
    validExploredStatesOneHot = oneHotEncoding(validExploredStates).to(device)  

    MovesToGo = net(validExploredStatesOneHot)
    
    targets = torch.zeros(exploredStates.shape[:2]).to(device)
    targets[validNextStates & ~goalNextStates] = MovesToGo.squeeze()

    targets[~validNextStates] = float("inf")

    targets = targets.min(1).values + 1
    targets[goalStates] = 0

    print(targets)

def oneHotEncoding(states):
    encodedStates = f.one_hot(states.view(-1, states.shape[1]**2).long(), -1).view(-1, states.shape[1]**4).float()
    return encodedStates



p = PuzzleN(15)

device = torch.device(0 if torch.cuda.is_available() else 'cpu')
puzzleNet = Net((p.N + 1) ** 2).to(device)

makeTrainingData(p, puzzleNet, device, 15, 2)