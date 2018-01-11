import numpy as np
import torch

class Normalize(object):
    '''Class containing normalization transformation for custom torch dataset
        Args:
                mean - mean of dataset examples
                sd - standard deviation of dataset examples
    '''
    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd
    def __call__(self, sample):
        individual = (sample["Individual"] - self.mean)/self.sd
        aggregate = (sample["Aggregate"] - self.mean)/self.sd
        return aggregate, individual
