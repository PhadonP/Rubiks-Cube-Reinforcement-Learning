from Node import Node
import time
import torch
from heapq import heapify, heappush, heappop

class BWAS():
    def search(self, scramble, depthPenalty):

        self.depthPenalty = depthPenalty
        self.open = heapq.heapify([])
        self.closed = set()

        root = Node(scramble, None, None, 0)
        self.open.heappush(root)
        self.closed.add(root)

        searchItr = 1
	    numNodesGenerated = 1
	    isSolved = False

        while not isSolved:
            
        


