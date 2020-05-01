import torch
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader

def makeTrainingData(environment, net, device, numStates, maxScramble):

    states, _ = environment.generateScrambles(numStates, maxScramble)
    goalStates = torch.all(states == environment.solvedState, 2)
    goalStates = goalStates.all(1).to(device)
    
    exploredStates, validNextStates, goalNextStates = environment.exploreNextStates(states)

    validExploredStates = exploredStates[validNextStates & ~goalNextStates]
    validExploredStatesOneHot = oneHotEncoding(validExploredStates).to(device).detach()

    MovesToGo = net(validExploredStatesOneHot)
    
    targets = torch.zeros(exploredStates.shape[:2]).to(device)
    targets[validNextStates & ~goalNextStates] = MovesToGo.squeeze()
    targets[goalNextStates] = 0
    targets[~validNextStates] = float("inf")

    targets = targets.min(1).values + 1

    targets[goalStates] = 0
    
    encodedStates = oneHotEncoding(states).to(device)

    return encodedStates, targets

def oneHotEncoding(states):
    rowLength = states.shape[1]
    boardSize = states.shape[1] ** 2
    states = states.view(-1, boardSize)
    _, indices = states.sort(1)
    encodedStates = torch.zeros(states.shape[0], boardSize, boardSize)
    encodedStates = encodedStates.scatter(2, indices.unsqueeze(-1),1).view(-1, boardSize, rowLength, rowLength).float()
    return encodedStates


class Puzzle15DataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train(net, device, loader, optimizer, lossLogger):
    #Set Network in train mode
    net.train()
    epochLoss = 0
    
    for i, (x, y) in enumerate(loader):
        
        #load images and labels to device
        x = x.to(device) # x is the image
        y = y.to(device) # y is the corresponding label
        
        #Forward pass of image through network and get output
        fx = net(x).squeeze()

        #Calculate loss using loss function
        loss = torch.nn.MSELoss()(fx, y)

        #Zero Gradents
        optimizer.zero_grad()
        #Backpropagate Gradents
        loss.backward()
        #Do a single optimization step
        optimizer.step()
        
        #create the cumulative sum of the loss and acc
        epochLoss += loss.item()

        #log the loss for plotting
        lossLogger.append(loss.item())
        
        print("TRAINING: | Iteration [%d/%d] | Loss %.2f |" %(i+1 ,len(loader) , loss.item()))
    
    #return the average loss and acc from the epoch as well as the logger array       
    return epochLoss / len(loader), lossLogger