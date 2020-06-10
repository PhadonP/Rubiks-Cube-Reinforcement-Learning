import torch
import config.config as config
import argparse
import os

from search.BWAS import batchedWeightedAStarSearch
from environment.CubeN import CubeN
from networks.getNetwork import getNetwork

from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS


if __name__ == "__main__":
    reqParser = reqparse.RequestParser()
    reqParser.add_argument("scramble", type=int,
                           action="append", help="Scramble to Solve")
    reqParser.add_argument("cubeSize", type=int, help="Size of Cube")

    app = Flask(__name__)
    api = Api(app)
    CORS(app)

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-n2", "--twoNet", required=True, help="Path of 2x2 Network", type=str
    )

    argParser.add_argument(
        "-n3", "--threeNet", required=True, help="Path of 3x3 Network", type=str
    )

    argParser.add_argument(
        "-c2", "--configTwo", help="2x2 Config File", type=str
    )

    argParser.add_argument(
        "-c3", "--configThree", help="3x3 Config File", type=str
    )

    args = argParser.parse_args()

    conf2 = config.Config(args.configTwo)
    conf3 = config.Config(args.configThree)

    loadPathTwo = args.twoNet
    loadPathThree = args.threeNet

    if not os.path.isfile(loadPathTwo):
        raise ValueError("No 2x2 Network Saved in this Path")
    if not os.path.isfile(loadPathThree):
        raise ValueError("No 3x3 Network Saved in this Path")

    env2 = CubeN(conf2.puzzleSize)
    net2 = getNetwork(conf2.puzzle, conf2.networkType)(conf2.puzzleSize)

    env3 = CubeN(conf3.puzzleSize)
    net3 = getNetwork(conf3.puzzle, conf3.networkType)(conf3.puzzleSize)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net2.to(device)

    net2.load_state_dict(torch.load(loadPathTwo))
    net2.eval()

    net3.to(device)

    net3.load_state_dict(torch.load(loadPathThree))
    net3.eval()

    class Solve(Resource):
        def post(self):
            args = reqParser.parse_args()
            scramble = torch.tensor(args.scramble, dtype=torch.uint8)
            print(scramble)

            if args.cubeSize == 3:
                (
                    moves,
                    numNodesGenerated,
                    searchItr,
                    isSolved,
                    solveTime,
                ) = batchedWeightedAStarSearch(
                    scramble,
                    conf3.depthWeight,
                    conf3.numParallel,
                    env3,
                    net3,
                    device,
                    conf3.maxSearchItr,
                )
            elif args.cubeSize == 2:
                (
                    moves,
                    numNodesGenerated,
                    searchItr,
                    isSolved,
                    solveTime,
                ) = batchedWeightedAStarSearch(
                    scramble,
                    conf2.depthWeight,
                    conf2.numParallel,
                    env2,
                    net2,
                    device,
                    conf2.maxSearchItr,
                )
            else:
                print("Invalid Cube Size")

            return {"isSolved": isSolved, "solveTime": solveTime, "solve": moves}

    api.add_resource(Solve, "/")

    app.run(debug=True)
