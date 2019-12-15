import torch
import random

class PuzzleN():
    def __init__(self, N):
        assert N in [3, 8, 15, 24, 35, 48], "N must be valid"
        self.N = N
        self.rowLength = int((N + 1) ** 0.5)
        self.actions = {"U" : [1, 0],
                        "R" : [0, -1],
                        "D" : [-1, 0],
                        "L" : [0, 1]
                        }
        
        self.state = self.getSolvedState()
        self.solvedState = self.getSolvedState()
        self.missing = [self.rowLength - 1, self.rowLength - 1]

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
        movingSquare = [sum(x) for x in zip(self.missing, self.actions[action])]
        if self.validAction(movingSquare):
            self.state[tuple(self.missing)], self.state[tuple(movingSquare)], = self.state[tuple(movingSquare)], 0
            self.missing = movingSquare

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