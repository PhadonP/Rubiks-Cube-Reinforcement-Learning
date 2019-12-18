import torch
import torch.nn.functional as f

def makeTrainingData(environment, net, device, numStates, maxScramble):

    states = environment.generateScrambles(numStates, maxScramble)
    goalStates = torch.all(states == environment.solvedState, 2)
    goalStates = goalStates.all(1).to(device)

    exploredStates, validNextStates, goalNextStates = environment.exploreNextStates(states)

    validExploredStates = exploredStates[validNextStates & ~goalNextStates]
    validExploredStatesOneHot = oneHotEncoding(validExploredStates).to(device).detach()

    MovesToGo = net(validExploredStatesOneHot)
    
    targets = torch.zeros(exploredStates.shape[:2]).to(device)
    targets[validNextStates & ~goalNextStates] = MovesToGo.squeeze()

    targets[~validNextStates] = float("inf")

    targets = targets.min(1).values + 1
    targets[goalStates] = 0
    
    encodedStates = oneHotEncoding(states).to(device).detach()

    return encodedStates, targets.unsqueeze(1)

def oneHotEncoding(states):
    encodedStates = f.one_hot(states.view(-1, states.shape[1]**2).long(), -1).view(-1, states.shape[1]**4).float()
    return encodedStates




def train(net, device, loader, optimizer, Loss_fun, loss_logger, acc_logger):
