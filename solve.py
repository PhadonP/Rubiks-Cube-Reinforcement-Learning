import config
import argparse
import os

from searchUtils.BWAS import batchedWeightedAStarSearch
from environment.PuzzleN import PuzzleN
from net import Net

import torch

if __name__ == "__main__":

    conf = config.Config("ini/15puzzleinitial.ini")
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--network", required=True, help="Path of Saved Network")
    args = parser.parse_args()
    
    loadPath = args.network

    if not os.path.isdir(loadPath):
        ValueError("No Network Saved in this Path")

    env = PuzzleN(conf.puzzleSize)
    
    device = torch.device(0 if torch.cuda.is_available() else 'cpu')

    net = Net(conf.puzzleSize + 1).to(device)
    net.load_state_dict(torch.load(loadPath)['net_state_dict'])
    net.eval()

    scramble = env.generateScramble(50)
    print(scramble)

    moves, numNodesGenerated, searchItr = batchedWeightedAStarSearch(scramble, conf.depthPenalty, conf.numParallel, env, net, device)

    print("Solved!")
    print("Moves are %s" % moves)
    print("Solve Length is %i" % len(moves)) 
    print("%i Nodes were generated" % numNodesGenerated)  
    print("There were %i search iterations" % searchItr) 

    