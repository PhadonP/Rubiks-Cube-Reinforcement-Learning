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

reqParser = reqparse.RequestParser()
reqParser.add_argument("scramble", type=int,
                       action="append", help="Scramble to Solve")
reqParser.add_argument("cubeSize", type=int, help="Size of Cube")

app = Flask(__name__)
api = Api(app)
CORS(app)

argParser = argparse.ArgumentParser()
argParser.add_argument(
    "-n", "--network", required=True, help="Path of Saved Network", type=str
)
argParser.add_argument(
    "-c", "--config", required=True, help="Path of Config File", type=str
)

args = argParser.parse_args()

conf = config.Config(args.config)

loadPath = args.network

if not os.path.isfile(loadPath):
    raise ValueError("No Network Saved in this Path")

env = CubeN(conf.puzzleSize)
net = getNetwork(conf.puzzle, conf.networkType)(conf.puzzleSize)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net.to(device)

net.load_state_dict(torch.load(loadPath))
net.eval()

heuristicFn = net


class Solve(Resource):
    def post(self):
        args = reqParser.parse_args()
        scramble = torch.tensor(args.scramble, dtype=torch.uint8)
        print(scramble)

        if cubeSize == 3:
            print("No 3x3 Network Yet")
            return

        (
            moves,
            numNodesGenerated,
            searchItr,
            isSolved,
            solveTime,
        ) = batchedWeightedAStarSearch(
            scramble,
            conf.depthWeight,
            conf.numParallel,
            env,
            heuristicFn,
            device,
            conf.maxSearchItr,
        )
        return {"isSolved": isSolved, "solveTime": solveTime, "solve": moves}


api.add_resource(Solve, "/")

if __name__ == "__main__":
    app.run(debug=True)
