import torch

class Node():
    def __init__(self, state, parent, parentMove, depth, cost, isSolved):

            self.depth = depth #Moves from root to this state
            self.cost = None #Addition of Heuristic and current path cost

            self.state = state
            self.parent = parent
            self.parentMove = parentMove
            self.isSolved = isSolved

            self.hashState = hash(self.state)

    def __hash__(self):
        return(self.hashState)
    
    def setCost(self):
        if self.heuristic:
            self.cost = self.heuristic + self.depth

    def __eq__(self, other):
        return torch.equal(self.state, other.state)

    def __ne__(self, other):
        return not self.__eq__(other)