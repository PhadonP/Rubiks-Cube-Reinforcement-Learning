import config
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

    if args.heuristicFunction == "net":
        heuristicFn = net
    elif args.heuristicFunction == "manhattan":
        heuristicFn = env.manhattanDistance
    else:
        raise ValueError("Invalid Heuristic Function")

    movesList = []
    numNodesGeneratedList = []
    searchItrList = []
    isSolvedList = []
    timeList = []
    for _ in range(2):

        scramble = env.generateScramble(args.scrambleDepth)

        (
            moves,
            numNodesGenerated,
            searchItr,
            isSolved,
            time,
        ) = batchedWeightedAStarSearch(
            scramble,
            conf.depthWeight,
            conf.numParallel,
            env,
            heuristicFn,
            device,
            conf.maxSearchItr,
        )

        if moves:
            movesList.append(len(moves))
                
        numNodesGeneratedList.append(numNodesGenerated)
        searchItrList.append(searchItrList)
        isSolvedList.append(isSolved)
        timeList.append(time)

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

    print("Average Move Count %.2f" % (sum(movesList) / len(movesList)))
    print(
        "Average Nodes Generated: %.2f"
        % (sum(numNodesGeneratedList) / len(numNodesGeneratedList))
    )
    print("Number Solved: %d" % isSolvedList.count(True))
    print("Average Time: %.2f seconds" % (sum(timeList) / len(timeList)))
