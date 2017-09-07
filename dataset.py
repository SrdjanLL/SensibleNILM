import torch
import torch.utils.data as torch_data
import data
import pandas as pd
import numpy as np
from parameters import *
import matplotlib.pyplot as plt
import random


class RefrigeratorDataSet(torch_data.Dataset):
    """REDD dataset, dataset made by MIT for NIALM, converter to Torch Tensor """

    def __init__(self, data_dir, transform=None):
        #self.homes = data.read_all_homes(data_dir)
        self.data_dir = data_dir

        #Reading two mains and refrigerator data from house1 (Nascimento used this data for training)
        aggregate1, individual1 = data.downsampled_channels(data_dir + "h1/channel_1.dat", data_dir + "h1/channel_2.dat", data_dir + "h1/channel_5.dat")

        #Conversion of data to 2D torch Tensors
        aggregate1,individual1 = data.convert_to_tensor(aggregate1, individual1, REFRIGERATOR_WINDOW_SIZE)


        #Reading two mains and refrigerator data from house2 (Nascimento used this data for testing)
        aggregate2, individual2 = data.downsampled_channels(data_dir + "h2/channel_1.dat", data_dir + "h2/channel_2.dat", data_dir + "h1/channel_9.dat")
        aggregate2, individual2 = data.convert_to_tensor(aggregate2, individual2, REFRIGERATOR_WINDOW_SIZE)

        #Reading two mains and refrigerator data from house3 (Nascimento used this data for training)
        aggregate3, individual3 = data.downsampled_channels(data_dir + "h3/channel_1.dat", data_dir + "h3/channel_2.dat", data_dir + "h1/channel_7.dat")
        aggregate3, individual3 = data.convert_to_tensor(aggregate3, individual3, REFRIGERATOR_WINDOW_SIZE)

        #Reading two mains and refrigerator data from house 6 (Nascimento used this data for testing)
        aggregate6, individual6 = data.downsampled_channels(data_dir + "h6/channel_1.dat", data_dir + "h6/channel_2.dat", data_dir + "h6/channel_8.dat")
        aggregate6, individual6 = data.convert_to_tensor(aggregate6, individual6, REFRIGERATOR_WINDOW_SIZE)

        self.aggregate = torch.cat((aggregate1, aggregate2, aggregate3, aggregate6), 0 )
        self.individual = torch.cat((individual1, individual2, individual3, individual6), 0)

    def __len__(self):
        return len(self.aggregate)

    def __getitem__(self, index):
	"""Geting one item from a dataset in format: [input, desired_output]"""
        if index >= 0 and index < self.__len__():
            #print(index, ": (", self.aggregate[index], "), (", self.individual[index], ").")
            return self.aggregate[index], self.individual[index]
        else:
            return None
