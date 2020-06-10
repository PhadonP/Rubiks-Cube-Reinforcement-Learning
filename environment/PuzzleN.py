import torch
import random
import numpy as np
import time


class PuzzleN:
    def __init__(self, N):
        assert N in [3, 8, 15, 24, 35, 48], "N must be valid"
        self.N = N
        self.rowLength = int((N + 1) ** 0.5)
        self.actions = {
            "U": torch.tensor([1, 0]),
            "R": torch.tensor([0, -1]),
            "D": torch.tensor([-1, 0]),
            "L": torch.tensor([0, 1]),
        }

        self.state = self.getSolvedState()
        self.solvedState = self.getSolvedState()

        self.manDistMat = self.generateManDistMat()

    def getSolvedState(self):
        state = []
        for i in range(self.rowLength):
            state.append(
                [n + self.rowLength * i for n in range(1, self.rowLength + 1)])
        state[-1][-1] = 0

        return torch.tensor(state, dtype=torch.uint8)

    def checkIfSolved(self, states):
        goals = torch.all(states == self.solvedState, 2)
        goals = goals.all(1)
        return goals

    def checkIfSolvedSingle(self, state):
        return torch.equal(state, self.solvedState)

    def nextState(self, states, actions):
        stateIdxs, missingY, missingX = torch.where(states == 0)
        missing = torch.stack((missingY, missingX), 1)
        movingSquare = missing + torch.stack(
            [self.actions[action] for action in actions]
        )

        movingSquare = torch.cat((stateIdxs.unsqueeze(1), movingSquare), 1)
        missing = torch.cat((stateIdxs.unsqueeze(1), missing), 1)

        invalids = missing[
            torch.any(
                (movingSquare[:, 1:] >= self.rowLength) | (
                    movingSquare[:, 1:] < 0), 1
            )
        ][:, 0]
        missing = missing[
            torch.all(
                (movingSquare[:, 1:] < self.rowLength) & (
                    movingSquare[:, 1:] >= 0), 1
            )
        ]
        movingSquare = movingSquare[
            torch.all(
                (movingSquare[:, 1:] < self.rowLength) & (
                    movingSquare[:, 1:] >= 0), 1
            )
        ]

        stateIdxs, missingY, missingX = missing[:,
                                                0], missing[:, 1], missing[:, 2]
        movingSquareY, movingSquareX = movingSquare[:, 1], movingSquare[:, 2]

        states[stateIdxs, missingY, missingX] = states[
            stateIdxs, movingSquareY, movingSquareX
        ]
        states[stateIdxs, movingSquareY, movingSquareX] = 0

        return states, stateIdxs, invalids

    def doAction(self, action, state=None):
        assert action in self.actions
        if state is None:
            state = self.state
        missing = torch.tensor(torch.where(state == 0))
        movingSquare = missing + self.actions[action]

        if self.validAction(movingSquare):
            state[tuple(missing)], state[tuple(movingSquare)], = (
                state[tuple(movingSquare)],
                0,
            )

        return state

    def validAction(self, movedSquare):
        return (
            0 <= movedSquare[0] < self.rowLength
            and 0 <= movedSquare[1] < self.rowLength
        )

    def generateScramble(self, noMoves):
        state = self.solvedState.clone()
        missing = [self.rowLength - 1, self.rowLength - 1]
        scramble = []
        movesDone = 0

        while movesDone < noMoves:
            randomMove = random.choice(list(self.actions.values()))
            movingSquare = [sum(x) for x in zip(missing, randomMove)]
            if self.validAction(movingSquare):
                movesDone += 1
                state[tuple(missing)], state[tuple(movingSquare)], = (
                    state[tuple(movingSquare)],
                    0,
                )
                missing = movingSquare

        return state

    def generateScrambles(self, numStates, maxScrambles, minScrambles=0):

        states = self.solvedState.repeat(numStates, 1, 1)
        scrambleNums = np.random.randint(minScrambles, maxScrambles + 1, numStates)
        numMoves = np.zeros(numStates)

        while np.max(numMoves < scrambleNums):

            poses = np.where((numMoves < scrambleNums))[0]

            subsetSize = max(len(poses) // len(self.actions), 1)

            poses = np.random.choice(poses, subsetSize)

            move = random.choice(list(self.actions.keys()))
            states[poses], valids, _ = self.nextState(states[poses], move)
            numMoves[poses[valids.cpu()]] += 1

        return states

    def exploreNextStates(self, states):
        nextStates = states.unsqueeze(1).repeat(1, len(self.actions), 1, 1)
        validStates = torch.tensor([True] * len(self.actions)).repeat(
            states.shape[0], 1
        )

        i = 0
        for action in self.actions:
            nextStates[:, i, :, :], _, invalids = self.nextState(
                nextStates[:, i, :, :], [action] * states.shape[0]
            )
            validStates[invalids, i] = False
            i += 1

        goals = self.checkIfSolved(
            nextStates.view(-1, self.rowLength, self.rowLength)
        ).view(-1, len(self.actions))

        return nextStates, validStates, goals

    def NextStateSpotToAction(self, i):
        if i == 0:
            return "U"
        elif i == 1:
            return "R"
        elif i == 2:
            return "D"
        elif i == 3:
            return "L"

    @staticmethod
    def oneHotEncoding(states):
        rowLength = states.shape[1]
        boardSize = states.shape[1] ** 2
        states = states.view(-1, boardSize)
        _, indices = states.sort(1)
        encodedStates = torch.zeros(states.shape[0], boardSize, boardSize)
        encodedStates = (
            encodedStates.scatter(2, indices.unsqueeze(-1), 1)
            .flatten(start_dim=1)
            .float()
        )

        return encodedStates

    def generateManDistMat(self):
        manDistMat = torch.zeros(self.N + 1, self.N + 1, requires_grad=False)

        for pos in range(self.N + 1):
            for num in range(self.N + 1):
                if pos == 0:
                    currRow = self.rowLength - 1
                    currCol = self.rowLength - 1
                else:
                    currRow = (pos - 1) // self.rowLength
                    currCol = (pos - 1) % self.rowLength

                solvedRow = (num - 1) // self.rowLength
                solvedCol = (num - 1) % self.rowLength
                if num != 0:
                    diffRows = abs(solvedRow - currRow)
                    diffCols = abs(solvedCol - currCol)
                else:
                    diffRows = 0
                    diffCols = 0

                if pos != 0:
                    manDistMat[pos - 1][num] = diffRows + diffCols
                else:
                    manDistMat[-1][num] = diffRows + diffCols

        return manDistMat

    def manhattanDistance(self, states):
        distance = torch.zeros(states.shape[0], dtype=torch.uint8)
        states = states.flatten(start_dim=1)

        for i, state in enumerate(states):
            distance[i] = sum(self.manDistMat.gather(
                1, state.unsqueeze(-1).long()))
        return distance
