from puzzleNEnv import PuzzleN
from net import Net
import torch
import torch.nn.functional as f
import trainUtils
import config

if __name__ == "__main__":

    conf = config.Config("ini/15puzzleinitial.ini")
    env = PuzzleN(conf.puzzleSize)
    
    device = torch.device(0 if torch.cuda.is_available() else 'cpu')
    net = Net((conf.puzzleSize + 1) ** 2).to(device)

    #for i in range(5):

    scrambles, targetMovesToGo = trainUtils.makeTrainingData(env, net, device, conf.numberOfScrambles, conf.scrambleDepth)
    
    