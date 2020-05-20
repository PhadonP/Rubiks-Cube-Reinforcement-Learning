import configparser
import time


class Config:
    def __init__(self, file_name):
        self.data = configparser.ConfigParser()

        if not self.data.read(file_name):
            raise ValueError("Config file %s not found" % file_name)

    @property
    def general(self):
        return self.data["general"]

    @property
    def train(self):
        return self.data["train"]

    @property
    def solve(self):
        return self.data["solve"]

    @property
    def puzzle(self):
        return self.general["puzzle"]

    @property
    def puzzleSize(self):
        return self.general.getint("puzzleSize")

    @property
    def numWorkers(self):
        return self.general.getint("numWorkers")

    @property
    def networkType(self):
        return self.general["networkType"]

    @property
    def numberOfScrambles(self):
        return self.train.getint("numberOfScrambles")

    @property
    def scrambleDepth(self):
        return self.train.getint("scrambleDepth")

    @property
    def batchSize(self):
        return self.train.getint("batchSize")

    @property
    def numEpochs(self):
        return self.train.getint("numEpochs")

    @property
    def lr(self):
        return self.train.getfloat("lr")

    @property
    def lrDecay(self):
        return self.train.getfloat("lrDecay")

    @property
    def weightDecay(self):
        return self.train.getfloat("weightDecay")

    @property
    def checkEpoch(self):
        return self.train.getint("checkEpoch")

    @property
    def lossThreshold(self):
        return self.train.getfloat("lossThreshold")

    @property
    def numTestScrambles(self):
        return self.train.getint("numTestScrambles")

    @property
    def testScrambleDepth(self):
        return self.train.getint("testScrambleDepth")

    @property
    def depthWeight(self):
        return self.solve.getfloat("depthWeight")

    @property
    def numParallel(self):
        return self.solve.getint("numParallel")

    @property
    def maxSearchItr(self):
        return self.solve.getint("maxSearchItr")

    def trainName(self, suffix=None):
        name = "%s-%d,ScrambleDepth-%d,Epochs-%d,lr-%f,%s" % (
            self.puzzle.capitalize()[:-1],
            self.puzzleSize,
            self.scrambleDepth,
            self.numEpochs,
            self.lr,
            self.networkType
        )
        if suffix:
            name += "," + suffix
        return name
