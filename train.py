import os
import time
import argparse
import config.config as config

from environment.getEnvironment import getEnvironment
from networks.getNetwork import getNetwork

import multiprocessing as mp
from training import trainUtils
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, help="Path of Config File", type=str
    )
    parser.add_argument("-n", "--name", default=None,
                        help="Name of Save", type=str)

    parser.add_argument("-nt", "--network", default=None,
                        help="Start with Existing Network", type=str)

    parser.add_argument("-mp", "--multiprocessing", default=False,
                        help="Turn on Multiprocessing", type=bool)

    args = parser.parse_args()

    conf = config.Config(args.config)

    env = getEnvironment(conf.puzzle)(conf.puzzleSize)

    netConstructor = getNetwork(conf.puzzle, conf.networkType)
    net = netConstructor(conf.puzzleSize)
    targetNet = netConstructor(conf.puzzleSize)
    solvingNet = netConstructor(conf.puzzleSize)

    name = conf.trainName(args.name)

    tb = SummaryWriter(comment=name)

    tbState = env.oneHotEncoding(env.generateScrambles(1, 0))
    tb.add_graph(net, tbState)

    if args.network:
        netSavePath = args.network
        if os.path.isfile(netSavePath):
            net.load_state_dict(torch.load(args.network))
            targetNet.load_state_dict(torch.load(args.network))
            print("Loading Network")
        else:
            raise ValueError("No Network Found")
    else:
        netSavePath = os.path.join("saves", name) + ".pt"

    resultsSavePath = os.path.join("trainresults", name) + ".csv"

    if os.path.exists(resultsSavePath):
        os.remove(resultsSavePath)

    if args.multiprocessing:
        procs = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net.to(device)
    targetNet.to(device)

    targetNet.load_state_dict(net.state_dict())

    optimizer = torch.optim.Adam(
        net.parameters(), lr=conf.lr, weight_decay=conf.weightDecay)
    scheduler = StepLR(optimizer, step_size=1,
                       gamma=conf.lrDecay)

    numEpochs = conf.numEpochs
    numWorkers = conf.numWorkers
    numberOfScrambles = conf.numberOfScrambles
    scrambleDepth = conf.scrambleDepth
    batchSize = conf.batchSize
    checkEpoch = conf.checkEpoch
    lossThreshold = conf.lossThreshold

    startTrainTime = time.time()

    for epoch in range(1, numEpochs + 1):

        if (epoch % 100 == 1):
            startPrepTime = time.time()

            dataIdx = 0
            scrambles, preparedData = trainUtils.prepareTrainingData(
                env, numberOfScrambles * 100, scrambleDepth)

            prepTime = time.time() - startPrepTime
            print("Preparation of Training Data time: %.3f seconds" % prepTime)

        startEpochTime = time.time()

        preparedChunk = [data[dataIdx*numberOfScrambles:(dataIdx + 1) *
                              numberOfScrambles] for data in preparedData]
        scramblesChunk = scrambles[dataIdx *
                                   numberOfScrambles:(dataIdx + 1)*numberOfScrambles]
        dataIdx += 1

        targets = trainUtils.makeTrainingData(
            env, preparedChunk, targetNet, device
        )

        scramblesDataSet = trainUtils.Puzzle15DataSet(scramblesChunk, targets)

        trainLoader = torch.utils.data.DataLoader(
            scramblesDataSet,
            batch_size=batchSize,
            shuffle=True,
            num_workers=numWorkers
        )

        meanLoss, meanValue = trainUtils.train(
            net, device, trainLoader, optimizer
        )

        scheduler.step()

        epochTime = time.time() - startEpochTime

        print("Epoch: %d/%d | Epoch Time: %.3f seconds" %
              (epoch, numEpochs, epochTime))

        tb.add_scalar('Loss', meanLoss, epoch)
        tb.add_scalar('Value', meanValue, epoch)
        tb.add_scalar('Time', epochTime, epoch)

        if epoch == 100:
            targetNet.load_state_dict(net.state_dict())

        if epoch % checkEpoch == 0:
            if meanLoss < lossThreshold:
                targetNet.load_state_dict(net.state_dict())
                print("Saving Model", flush=True)
                torch.save(net.state_dict(), netSavePath)
            else:
                print("Loss is too high, unable to update target network", flush=True)

        if (epoch % 100 == 0 and epoch < 1000 and numEpochs < 50000) or (epoch % 1000 == 0 and epoch < 10000) or epoch % 10000 == 0:

            tb.flush()
            print("Testing Network", flush=True)

            if args.multiprocessing:

                solvingNet.load_state_dict(net.state_dict())

                p = mp.Process(target=trainUtils.test, args=(
                    epoch, env, solvingNet, "cpu", conf, resultsSavePath, False))
                p.start()
                procs.append(p)

            else:
                trainUtils.test(epoch, env, net, device,
                                conf, resultsSavePath, True)

    trainTime = time.time() - startTrainTime

    print(
        "Training Time is %i hours, %i minutes and %i seconds"
        % (trainTime / 3600, trainTime / 60 % 60, trainTime % 60)
    )

    if args.multiprocessing:
        for p in procs:
            p.join()

    tb.close()
