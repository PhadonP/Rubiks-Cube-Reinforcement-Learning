import timeit
import torch
import random
import numpy as np


class PuzzleN():
    def __init__(self, N):
        assert N in [3, 8, 15, 24, 35, 48], "N must be valid"
        self.N = N
        self.rowLength = int((N + 1) ** 0.5)
        self.actions = {"U" : torch.tensor([1, 0]),
                        "R" : torch.tensor([0, -1]),
                        "D" : torch.tensor([-1, 0]),
                        "L" : torch.tensor([0, 1])
                        }
        
        self.state = self.getSolvedState()
        self.solvedState = self.getSolvedState()

    def getSolvedState(self):
        state = []
        for i in range(self.rowLength):
            state.append([n + self.rowLength * i for n in range(1, self.rowLength + 1)])
        state[-1][-1] = 0

        if self.N <= 15:
            dataType = torch.uint8
        else:
            dataType = torch.int
        return torch.tensor(state, dtype = dataType)
    
    def checkIfSolved(self):
        return torch.equal(self.state, self.solvedState)
    
    def getState(self):
        return self.state
    
    def doAction(self, action):
        assert action in self.actions
        missing = torch.tensor(torch.where(self.state == 0))
        movingSquare = missing + self.actions[action]

        if self.validAction(movingSquare):
            self.state[tuple(missing)], self.state[tuple(movingSquare)], = self.state[tuple(movingSquare)], 0

    def validAction(self, movedSquare):
        return 0 <= movedSquare[0] < self.rowLength and 0 <= movedSquare[1] < self.rowLength
    
    def generateScramble(self, noMoves):
        state = self.getSolvedState()
        missing = [self.rowLength - 1, self.rowLength - 1]
        scramble = []
        movesDone = 0
        passedMove = None
        
        while movesDone < noMoves:
            randomMove = random.choice(list(self.actions.values()))
            if randomMove == passedMove:
                continue
            movingSquare = [sum(x) for x in zip(missing, randomMove)]
            if self.validAction(movingSquare):
                passedMove = randomMove
                movesDone += 1
                state[tuple(missing)], state[tuple(movingSquare)], = state[tuple(movingSquare)], 0
                missing = movingSquare
        
        return state

    def nextState(self, states, actions):
        stateIdxs, missingY, missingX = torch.where(states == 0)
        missing = torch.stack((missingY, missingX), 1)
        movingSquare = missing + torch.stack([self.actions[action] for action in actions])
        
        movingSquare = torch.cat((stateIdxs.unsqueeze(1), movingSquare),1)
        missing = torch.cat((stateIdxs.unsqueeze(1), missing),1)
        missing = missing[torch.all(movingSquare[:,1:] < self.rowLength,1)]
        movingSquare = movingSquare[torch.all(movingSquare[:,1:] < self.rowLength,1)]
        missing = missing[torch.all(movingSquare[:,1:] >= 0,1)]
        movingSquare = movingSquare[torch.all(movingSquare[:,1:] >= 0,1)]

        stateIdxs = missing[:,0]
        missingY = missing[:,1]
        missingX = missing[:,2]
        movingSquareY = movingSquare[:,1]
        movingSquareX = movingSquare[:,2]
        
        states[stateIdxs, missingY, missingX] = states[stateIdxs, movingSquareY, movingSquareX]
        states[stateIdxs, movingSquareY, movingSquareX] = 0
        
        return states, stateIdxs
    
    def generateScrambles(self, numStates, scrambleRange):
        scrambs = range(1,scrambleRange+1)
        states = self.solvedState.repeat(numStates, 1, 1)

        scrambleNums = np.random.choice(scrambs,numStates)
        numMoves = np.zeros(numStates)

        while np.max(numMoves < scrambleNums):

            poses = np.where((numMoves < scrambleNums))[0]

            subsetSize = max(len(poses)//len(self.actions),1)
            
            poses = np.random.choice(poses,subsetSize)

            move = random.choice(list(self.actions.keys()))
            states[poses], valids = self.nextState(states[poses], move)
            numMoves[poses[valids]] = numMoves[poses[valids]] + 1

        return(states,scrambleNums)


p = PuzzleN(15)

print(p.generateScrambles(100, 5))
