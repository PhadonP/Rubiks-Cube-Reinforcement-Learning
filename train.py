import os
import time
import argparse
import config.config as config
import torch.multiprocessing as mp

from environment.PuzzleN import PuzzleN
from environment.CubeN import CubeN
from networks.PuzzleNet import PuzzleNet
from networks.CubeNet import CubeNet

from training import trainUtils
import torch
from torch.optim.lr_scheduler import StepLR

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, help="Path of Config File", type=str
    )
    parser.add_argument("-n", "--name", default=None, help="Name of Save", type=str)

    args = parser.parse_args()

    conf = config.Config(args.config)

    if conf.puzzle == "puzzleN":
        env = PuzzleN(conf.puzzleSize)
        net = PuzzleNet(conf.puzzleSize)
        targetNet = PuzzleNet(conf.puzzleSize)
    elif conf.puzzle == "cubeN":
        env = CubeN(conf.puzzleSize)
        net = CubeNet(conf.puzzleSize)
        targetNet = CubeNet(conf.puzzleSize)
    else:
        raise ValueError("Invalid Puzzle")

    name = conf.trainName(args.name)

    savePath = os.path.join("saves", name) + ".pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numGPUs = torch.cuda.device_count()
    numProcs = conf.numProcs

    print("Using %d GPU(s)" % numGPUs)

    if numGPUs > 1:
        net = torch.nn.DataParallel(net)

    net.to(device)
    targetNet.to(device)

    mp.set_start_method("spawn")
    targetNet.share_memory()

    for targetParam, param in zip(net.parameters(), targetNet.parameters()):
        targetParam.data.copy_(param)

    numWorkers = conf.numWorkers

    optimizer = torch.optim.Adam(net.parameters(), lr=conf.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=conf.lrDecay)

    numEpochs = conf.numEpochs
    tau = conf.tau

    lossLogger = []

    startTrainTime = time.time()

    for epoch in range(1, numEpochs + 1):

        genProcesses = []
        genQueue = mp.Queue()

        startGenTime = time.time()

        for _ in range(numProcs):
            p = mp.Process(
                target=trainUtils.makeTrainingData,
                args=(
                    env,
                    targetNet,
                    device,
                    conf.numberOfScrambles // numProcs,
                    conf.scrambleDepth,
                    genQueue,
                ),
            )
            genProcesses.append(p)
            p.start()

        scrambles = []
        targets = []
        procsFinished = 0

        while procsFinished < numProcs:
            if not genQueue.empty():
                scrambleSet, targetSet = genQueue.get()
                scrambles.append(scrambleSet)
                targets.append(targetSet)
                procsFinished += 1

        for p in genProcesses:
            p.join()

        scrambles = torch.cat(scrambles)
        targets = torch.cat(targets)

        # scrambles, targets = trainUtils.makeTrainingData(
        #     env, targetNet, device, conf.numberOfScrambles, conf.scrambleDepth
        # )

        genTime = time.time() - startGenTime

        print("Generation time is %.3f seconds" % genTime)

        scramblesDataSet = trainUtils.Puzzle15DataSet(scrambles, targets)

        trainLoader = torch.utils.data.DataLoader(
            scramblesDataSet,
            batch_size=conf.batchSize,
            shuffle=True,
            num_workers=numWorkers,
        )

        avgLoss, lossLogger = trainUtils.train(
            net, device, trainLoader, optimizer, lossLogger
        )

        scheduler.step()

        for targetParam, param in zip(targetNet.parameters(), net.parameters()):
            targetParam.data.copy_(tau * param + (1 - tau) * targetParam)

        print("Epoch: %d/%d" % (epoch, numEpochs))

        epochTime = time.time()

        print("Epoch time: %.3f seconds" % (epochTime - startGenTime))

        if epoch % 10 == 0:
            print("Saving Model")
            torch.save(net.state_dict(), savePath)

    trainTime = time.time() - startTrainTime

    print(
        "Training Time is %i hours, %i minutes and %i seconds"
        % (trainTime / 3600, trainTime / 60 % 60, trainTime % 60)
    )
