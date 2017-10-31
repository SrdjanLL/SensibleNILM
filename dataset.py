import torch
import torch.utils.data as torch_data
import data
import pandas as pd
import numpy as np
from parameters import *
import matplotlib.pyplot as plt
import random

class REDDCleanDataset(torch_data.Dataset):
    def __init__(self, data_dir, transform=None, appliance='Refrigerator', window_size=REFRIGERATOR_WINDOW_SIZE, test=False, proportion=[1,1], threshold=10):
        self.data_dir = data_dir
        self.appliance = appliance
        self.window_size = window_size
        self.transform = transform
        if not test:
            self.data = data.generate_clean_data2(self.data_dir, self.appliance, self.window_size, threshold, proportion)
        else:
            self.data = data.generate_clean_test_data(self.data_dir, self.appliance, self.window_size)

    def get_mean_and_std(self):
        array = self.data[0]
        #array = np.reshape(array, (1, -1))
        print('Getting mean and sd. . .')
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

class REDDDataset(torch_data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def get_mean_and_std(self):
        array = self.data[0]
        #array = np.reshape(array, (1, -1))
        print('Getting mean and sd. . .')
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
