import numpy as np
import torch

class Normalize(object):
    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd
    def __call__(self, sample):
        individual = (sample["Individual"] - self.mean)/self.sd
        aggregate = (sample["Aggregate"] - self.mean)/self.sd
        return aggregate, individual
