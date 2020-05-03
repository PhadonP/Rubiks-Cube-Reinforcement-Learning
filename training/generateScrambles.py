import time
import torch
from environment.PuzzleN import PuzzleN


def generateScrambles(environment, numStates, scrambleDepth, filePathQueue):
    while not filePathQueue.empty():

        filePath = filePathQueue.get()
