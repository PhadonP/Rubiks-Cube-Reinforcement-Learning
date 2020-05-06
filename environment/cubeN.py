import torch
import random
import numpy as np


class CubeN:
    def __init__(self, N):
        assert N in [2, 3, 4, 5], "N must be valid"
        self.N = N
        actions = ["U", "U'", "F", "F'", "L", "L'", "D", "D'", "B", "B'", "R", "R'"]
        self.actions = {key: i for i, key in enumerate(actions)}

        self.state = self.getSolvedState()
        self.solvedState = self.getSolvedState()
        self.adjIdx = self.genAdjIdx()
        self.nextStateMat = self.genNextStateMat()

    def getSolvedState(self):
        state = torch.zeros(self.N ** 2 * 6, dtype=torch.uint8)
        for i in range(6):
            state[self.N ** 2 * i : self.N ** 2 * (i + 1)] = i

        return state

    def checkIfSolved(self, states):
        goals = torch.all(states == self.solvedState, 3)
        goals = goals.all(2)
        return goals

    def checkIfSolvedSingle(self, state):
        return torch.equal(state, self.solvedState)

    def genNextStateMat(self):
        solvedCube = np.arange(self.N ** 2 * 6)

        nextStateMat = np.tile(solvedCube, (len(self.actions), 1))

        for actIndex in range(len(self.actions)):
            # Rotate Face
            # faceToRotate 0: U, 1: F, 2: L, 3: D, 4: B, 5: R

            faceToRotate = actIndex // 2

            rotatedCube = solvedCube.copy()

            face = np.arange(
                faceToRotate * self.N ** 2, (faceToRotate + 1) * self.N ** 2
            ).reshape(self.N, self.N)

            adjPieces = rotatedCube[self.adjIdx[faceToRotate]]

            if actIndex % 2 == 0:
                rotatedFace = np.rot90(face, 3)
                adjPieces = np.roll(adjPieces, self.N)
            else:
                rotatedFace = np.rot90(face, 1)
                adjPieces = np.roll(adjPieces, -self.N)

            rotatedCube[
                faceToRotate * self.N ** 2 : (faceToRotate + 1) * self.N ** 2
            ] = rotatedFace.flatten()

            rotatedCube[self.adjIdx[faceToRotate]] = adjPieces
            nextStateMat[actIndex] = rotatedCube

        return nextStateMat

    def genAdjIdx(self):
        adjIdx = np.zeros((6, 4 * self.N), dtype=int)
        multiIdx = []
        for i in range(6):
            faces = [(i + j) % 6 for j in range(6) if j not in [0, 3]]
            if i % 2:
                faces = [faces[0]] + faces[1:][::-1]
            faces = np.array(faces).repeat(self.N)

            if i == 0:  # U
                # F L B R
                positionsRow = np.zeros(4 * self.N)
                positionsCol = np.tile(np.arange(self.N - 1, -1, -1), 4)

            if i == 1:  # F

                # L U R D
                positionsRow = np.array(
                    [
                        np.arange(self.N - 1, -1, -1),  # L
                        np.ones(self.N) * (self.N - 1),  # U
                        np.arange(self.N),  # R
                        np.zeros(self.N),  # D
                    ]
                )
                positionsCol = np.array(
                    [
                        np.ones(self.N) * (self.N - 1),  # L
                        np.arange(self.N),  # U
                        np.zeros(self.N),  # R
                        np.arange(self.N - 1, -1, -1),  # D
                    ]
                )
            if i == 2:  # L
                # D B U F
                positionsRow = np.array(
                    [
                        np.arange(self.N),  # D
                        np.arange(self.N - 1, -1, -1),  # B
                        np.arange(self.N),  # U
                        np.arange(self.N),  # F
                    ]
                )
                positionsCol = np.array(
                    [
                        np.zeros(self.N),  # D
                        np.ones(self.N) * (self.N - 1),  # B
                        np.zeros(self.N),  # U
                        np.zeros(self.N),  # F
                    ]
                )

            if i == 3:  # D
                # B L F R
                positionsRow = positionsRow = np.ones(4 * self.N) * (self.N - 1)
                positionsCol = positionsCol = np.tile(np.arange(self.N), 4)

            if i == 4:  # B
                # R U L D
                positionsRow = np.array(
                    [
                        np.arange(self.N - 1, -1, -1),  # R
                        np.zeros(self.N),  # U
                        np.arange(self.N),  # L
                        np.ones(self.N) * (self.N - 1),  # D
                    ]
                )
                positionsCol = np.array(
                    [
                        np.ones(self.N) * (self.N - 1),  # R
                        np.arange(self.N - 1, -1, -1),  # U
                        np.zeros(self.N),  # L
                        np.arange(self.N),  # D
                    ]
                )
            if i == 5:  # R
                # U B D F

                positionsRow = np.array(
                    [
                        np.arange(self.N - 1, -1, -1),  # U
                        np.arange(self.N),  # B
                        np.arange(self.N - 1, -1, -1),  # D
                        np.arange(self.N - 1, -1, -1),  # F
                    ]
                )
                positionsCol = np.array(
                    [
                        np.ones(self.N) * (self.N - 1),  # U
                        np.zeros(self.N),  # B
                        np.ones(self.N) * (self.N - 1),  # D
                        np.ones(self.N) * (self.N - 1),  # F
                    ]
                )
            positionsRow = positionsRow.flatten()
            positionsCol = positionsCol.flatten()

            multiIdx = np.stack(
                (faces.astype(int), positionsRow.astype(int), positionsCol.astype(int))
            )

            adjIdx[i] = np.ravel_multi_index(multiIdx, (6, self.N, self.N))

        return adjIdx

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
                (movingSquare[:, 1:] >= self.rowLength) | (movingSquare[:, 1:] < 0), 1
            )
        ][:, 0]
        missing = missing[
            torch.all(
                (movingSquare[:, 1:] < self.rowLength) & (movingSquare[:, 1:] >= 0), 1
            )
        ]
        movingSquare = movingSquare[
            torch.all(
                (movingSquare[:, 1:] < self.rowLength) & (movingSquare[:, 1:] >= 0), 1
            )
        ]

        stateIdxs, missingY, missingX = missing[:, 0], missing[:, 1], missing[:, 2]
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

        state = state[self.nextStateMat[self.actions[action]]]

        return state

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

    def generateScrambles(self, numStates, scrambleRange):
        scrambs = range(1, scrambleRange + 1)
        states = self.solvedState.repeat(numStates, 1, 1)

        scrambleNums = np.random.choice(scrambs, numStates)
        numMoves = np.zeros(numStates)

        while np.max(numMoves < scrambleNums):

            poses = np.where((numMoves < scrambleNums))[0]

            subsetSize = max(len(poses) // len(self.actions), 1)

            poses = np.random.choice(poses, subsetSize)

            move = random.choice(list(self.actions.keys()))
            states[poses], valids, _ = self.nextState(states[poses], move)
            numMoves[poses[valids]] = numMoves[poses[valids]] + 1

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

        goals = self.checkIfSolved(nextStates)

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
            .view(-1, boardSize, rowLength, rowLength)
            .float()
        )
        return encodedStates


cube3 = CubeN(3)

cube3.state = cube3.doAction("U'")
cube3.state = cube3.doAction("D'")
cube3.state = cube3.doAction("F'")
cube3.state = cube3.doAction("R'")
cube3.state = cube3.doAction("B")
cube3.state = cube3.doAction("U")
cube3.state = cube3.doAction("R")
cube3.state = cube3.doAction("L'")
cube3.state = cube3.doAction("B'")
cube3.state = cube3.doAction("L")


print(cube3.state.reshape(6, 3, 3))
