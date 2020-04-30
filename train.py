from puzzleNEnv import PuzzleN
from net import Net
import torch

import trainUtils
import config
import multiprocessing
import torch

if __name__ == "__main__":

    conf = config.Config("ini/15puzzleinitial.ini")
    env = PuzzleN(conf.puzzleSize)
    
    device = torch.device(0 if torch.cuda.is_available() else 'cpu')
    net = Net(conf.puzzleSize + 1).to(device)
    numWorkers = multiprocessing.cpu_count()
    optimizer = torch.optim.Adam(net.parameters(), lr = conf.lr)
    lossLogger = []
    accLogger = []

    scrambles, targetMovesToGo = trainUtils.makeTrainingData(env, net, device,
        conf.numberOfScrambles, conf.scrambleDepth)
    
    scramblesDataSet = trainUtils.Puzzle15DataSet(scrambles, targetMovesToGo)

    trainloader = torch.utils.data.DataLoader(scramblesDataSet,
        batch_size=conf.batchSize, shuffle=True, num_workers=0)

    avgLoss, avgAcc, lossLogger, accLogger = trainUtils.train(net, device, trainloader, 
        optimizer, lossLogger, accLogger)
    



    
    