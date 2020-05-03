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
    def puzzleSize(self):
        return self.general.getint("puzzleSize")

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
    def tau(self):
        return self.train.getfloat("tau")

    @property
    def depthPenalty(self):
        return self.solve.getfloat("depthPenalty")

    @property
    def numParallel(self):
        return self.solve.getint("numParallel")

    @property
    def maxSearchItr(self):
        return self.solve.getint("maxSearchItr")

    def trainName(self, suffix=None):
        name = "Puzzle-%d,ScrambleDepth=%d" % (self.puzzleSize, self.scrambleDepth)
        if suffix:
            name += "," + suffix
        return name
