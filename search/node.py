import torch


class Node:
    __slots__ = [
        "depth",
        "cost",
        "state",
        "parent",
        "parentMove",
        "isSolved",
        "hashState",
    ]

    def __init__(self, state, parent, parentMove, depth, cost, isSolved):

        self.depth = depth  # Moves from root to this state
        self.cost = None  # Addition of Heuristic and current path cost

        self.state = state
        self.parent = parent
        self.parentMove = parentMove
        self.isSolved = isSolved

        self.hashState = hash(self.state.numpy().data.tobytes())

    def __hash__(self):
        return self.hashState
