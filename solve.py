impors config
import argparse
import os

from search.BWAS import batchedWeightedAStarSearch
from environment.PuzzleN import PuzzleN
from net import Net

import torch

if __name__ == "__main__":

    conf = config.Config("ini/15puzzleinitial.ini")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--network", required=True, help="Path of Saved Network", type=str
    )
    parser.add_argument(
        "-s", "--scrambleDepth", required=True, help="Depth of Scramble", type=int
    )
    parser.add_argument(
        "-hf", "--heuristicFunction", required=True, help="net or manhattan", type=str
    )

    args = parser.parse_args()

    loadPath = args.network

    if not os.path.isfile(loadPath):
        raise ValueError("No Network Saved in this Path")

    env = PuzzleN(conf.puzzleSize)

    device = torch.device(0 if torch.cuda.is_available() else "cpu")

    net = Net(conf.puzzleSize + 1).to(device)
    net.load_state_dict(torch.load(loadPath)["net_state_dict"])
    net.eval()

    scramble = env.generateScramble(args.scrambleDepth)

    if args.heuristicFunction == "net":
        heuristicFn = net
    elif args.heuristicFunction == "manhattan":
        heuristicFn = env.manhattanDistance
    else:
        raise ValueError("Invalid Heuristic Function")

    moves, numNodesGenerated, searchItr, isSolved, time = batchedWeightedAStarSearch(
        scramble,
        conf.depthPenalty,
        conf.numParallel,
        env,
        heuristicFn,
        device,
        conf.maxSearchItr,
    )

    if isSolved:
        print("Solved!")
        print(scramble)
        print("Moves are %s" % "".join(moves))
        print("Solve Length is %i" % len(moves))
        print("%i Nodes were generated" % numNodesGenerated)
        print("There were %i search iterations" % searchItr)
        print("Time of Solve is %.3f seconds" % time)
    else:
        print(scramble)
        print("Max Search Iterations Reached")
        print("%i Nodes were generated" % numNodesGenerated)
        print("Search time was %.3f seconds" % time)
