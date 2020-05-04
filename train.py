import os
import time
import logging
import config
import multiprocessing

from environment.PuzzleN import PuzzleN
from net import Net
import torch

from training import trainUtils
import torch

from tensorboardX import SummaryWriter

if __name__ == "__main__":

    log = logging.getLogger("train")
    logging.basicConfig(
        format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO
    )
    conf = config.Config("ini/15puzzleinitial.ini")
    env = PuzzleN(conf.puzzleSize)

    name = conf.trainName()

    writer = SummaryWriter(comment="-" + name)
    savePath = os.path.join("saves", name) + ".pt"

    device = torch.device(0 if torch.cuda.is_available() else "cpu")

    net = Net(conf.puzzleSize + 1).to(device)
    targetNet = Net(conf.puzzleSize + 1).to(device)

    for targetParam, param in zip(net.parameters(), targetNet.parameters()):
        targetParam.data.copy_(param)

    numWorkers = conf.numWorkers
    numProcs = conf.numProcs

    optimizer = torch.optim.Adam(net.parameters(), lr=conf.lr)
    numEpochs = conf.numEpochs
    tau = conf.tau

    lossLogger = []

    startTrainTime = time.time()

    for epoch in range(1, numEpochs + 1):

        genProcesses = []

        for _ in range(numProcs):
            p = multiprocessing.Process(
                trainUtils.makeTrainingData,
                args=(
                    env,
                    targetNet,
                    device,
                    conf.numberOfScrambles // numProcs,
                    conf.scrambleDepth,
                ),
            )
            genProcesses.append(p)
            p.start()

        # scrambles, targetMovesToGo = trainUtils.makeTrainingData(
        #     env, targetNet, device, conf.numberOfScrambles, conf.scrambleDepth
        # )

        scramblesDataSet = trainUtils.Puzzle15DataSet(scrambles, targetMovesToGo)

        trainLoader = torch.utils.data.DataLoader(
            scramblesDataSet,
            batch_size=conf.batchSize,
            shuffle=True,
            num_workers=numWorkers,
        )

        avgLoss, lossLogger = trainUtils.train(
            net, device, trainLoader, optimizer, lossLogger
        )

        for targetParam, param in zip(targetNet.parameters(), net.parameters()):
            targetParam.data.copy_(tau * param + (1 - tau) * targetParam)

        print("Epoch: %d/%d" % (epoch, numEpochs))

        if epoch % 10 == 0:
            print("Saving Model")
            torch.save(net.state_dict(), savePath)

    trainTime = time.time() - startTrainTime

    print(
        "Training Time is %i hours, %i minutes and %i seconds"
        % (trainTime / 3600, trainTime / 60 % 60, trainTime % 60)
    )
