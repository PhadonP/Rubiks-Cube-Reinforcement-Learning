import os
import time
import argparse
import config
import logging

from environment.PuzzleN import PuzzleN
from net import Net
import torch

import trainUtils
import multiprocessing
import torch

from tensorboardX import SummaryWriter

if __name__ == "__main__":

    log = logging.getLogger("train")
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ini", required=True, help="Ini file to use for this run")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    conf = config.Config(args.ini)
    env = PuzzleN(conf.puzzleSize)

    name = conf.trainName(suffix=args.name)
    
    writer = SummaryWriter(comment="-" + name)
    savePath = os.path.join("saves", name)
    os.makedirs(savePath)
    
    device = torch.device(0 if torch.cuda.is_available() else 'cpu')

    net = Net(conf.puzzleSize + 1).to(device)
    targetNet = Net(conf.puzzleSize + 1).to(device)

    for targetParam, param in zip(net.parameters(), targetNet.parameters()):
        targetParam.data.copy_(param)

    numWorkers = multiprocessing.cpu_count()
    optimizer = torch.optim.Adam(net.parameters(), lr = conf.lr)
    numEpochs = conf.numEpochs
    tau = conf.tau
    
    lossLogger = []

    for epoch in range(numEpochs):

        scrambles, targetMovesToGo = trainUtils.makeTrainingData(env, targetNet, device,
            conf.numberOfScrambles, conf.scrambleDepth)
        
        scramblesDataSet = trainUtils.Puzzle15DataSet(scrambles, targetMovesToGo)

        trainLoader = torch.utils.data.DataLoader(scramblesDataSet,
            batch_size=conf.batchSize, shuffle=True, num_workers=0)

        avgLoss, lossLogger = trainUtils.train(net, device, trainLoader, optimizer, lossLogger)

        print("Epoch: %d" %(epoch))

        for targetParam, param in zip(targetNet.parameters(), net.parameters()):
            targetParam.data.copy_(tau * param + (1 - tau) * targetParam)

        if epoch % 10 == 0:
            print("Saving Model")
            torch.save({
                'epoch':                  epoch,
                'net_state_dict':         net.state_dict(),
                'targetNet_state_dict':   targetNet.state_dict(),
                'optimizer_state_dict':   optimizer.state_dict(),
                'loss_logger':            lossLogger
            }, savePath + ".pt")
        
        if epoch % 100 == 0:

            scrambles, numMoves = env.generateScrambles(20, conf.scrambleDepth)
            oneHot = env.oneHotEncoding(scrambles).to(device).detach()
            print(net(oneHot).squeeze())
            print(numMoves)
            print(scrambles)

        
        


    
    



    
    