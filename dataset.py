import torch
import torch.utils.data as torch_data
import pandas as pd
import numpy as np
from refit_parameters import *
import math

class RefitDataset(torch_data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def init_transformation(self, transform):
        if not self.transform:
            self.transform = transform
        else:
            print('Transformations are already predefined and you cannot initialize other transformations')

    def __getitem__(self, index):
        aggregate, iam = self.data[0][index], self.data[1][index]
        if self.transform:
            sample = {}
            sample['Aggregate'] = aggregate
            sample['Individual'] = iam
            aggregate, iam = self.transform(sample)
        aggregate, iam = torch.from_numpy(aggregate).double(), torch.from_numpy(iam).double()
        return aggregate, iam


class REDDDataset(torch_data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def get_mean_and_std(self):
        array = self.data[0]
        #array = np.reshape(array, (1, -1))
        return array.mean(), array.std()

    def __len__(self):
        return len(self.data[0])

    def init_transformation(self, transform):
        if not self.transform:
            self.transform = transform
        else:
            print('Transformations are already predefined and you cannot initialize other transformations')

    def __getitem__(self, index):
        aggregate, iam = self.data[0][index], self.data[1][index]
        if self.transform:
            sample = {}
            sample['Aggregate'] = aggregate
            sample['Individual'] = iam
            aggregate, iam = self.transform(sample)
        aggregate, iam = torch.from_numpy(aggregate), torch.from_numpy(iam)
return aggregate, iam
