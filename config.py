import configparser

class Config():
    def __init__(self, file_name):
        self.data = configparser.ConfigParser()

        if not self.data.read(file_name):
            raise ValueError("Config file %s not found" % file_name)

    @property
    def general(self):
        return self.data['general']

    @property
    def train(self):
        return self.data['train']

    @property
    def puzzleSize(self):
        return self.general.getint('puzzleSize')
    
    @property
    def numberOfScrambles(self):
        return self.train.getint('numberOfScrambles')
    
    @property
    def scrambleDepth(self):
        return self.train.getint('scrambleDepth')

    @property
    def batchSize(self):
        return self.train.getint('batchSize')
    
    @property
    def lr(self):
        return self.train.getfloat('lr')
    

