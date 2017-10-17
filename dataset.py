import torch
import torch.utils.data as torch_data
import data
import pandas as pd
import numpy as np
from parameters import *
import matplotlib.pyplot as plt
import random


class RefrigeratorREDDDataSet(torch_data.Dataset):
    """REDD dataset, dataset made by MIT for NIALM, converter to Torch Tensor """

    def __init__(self, data_dir, transform=None, type=None):
        self.data_dir = data_dir
        self.transform = transform
        #Reading two mains and refrigerator data from house1 (Nascimento used this data for training)
        aggregate1, individual1 = data.downsampled_channels(data_dir + "h1/channel_1.dat", data_dir + "h1/channel_2.dat", data_dir + "h1/channel_5.dat")

        #Conversion of data to 2D torch Tensors
        if type !="Overlap":
            aggregate1, individual1 = data.convert_to_tensor(aggregate1, individual1, REFRIGERATOR_WINDOW_SIZE)
        else:
            aggregate1, individual1 = data.convert_to_tensor_overlap(aggregate1, individual1, REFRIGERATOR_WINDOW_SIZE)

        #Reading two mains and refrigerator data from house2 (Nascimento used this data for testing)
        aggregate2, individual2 = data.downsampled_channels(data_dir + "h2/channel_1.dat", data_dir + "h2/channel_2.dat", data_dir + "h1/channel_9.dat")
        if type !="Overlap":
            aggregate2, individual2 = data.convert_to_tensor(aggregate2, individual2, REFRIGERATOR_WINDOW_SIZE)
        else:
            aggregate2, individual2 = data.convert_to_tensor_overlap(aggregate2, individual2, REFRIGERATOR_WINDOW_SIZE)
        #Reading two mains and refrigerator data from house3 (Nascimento used this data for training)
        aggregate3, individual3 = data.downsampled_channels(data_dir + "h3/channel_1.dat", data_dir + "h3/channel_2.dat", data_dir + "h1/channel_7.dat")
        if type !="Overlap":
            aggregate3, individual3 = data.convert_to_tensor(aggregate3, individual3, REFRIGERATOR_WINDOW_SIZE)
        else:
            aggregate3, individual3 = data.convert_to_tensor_overlap(aggregate3, individual3, REFRIGERATOR_WINDOW_SIZE)

        #Reading two mains and refrigerator data from house 6 (Nascimento used this data for testing)
        aggregate6, individual6 = data.downsampled_channels(data_dir + "h6/channel_1.dat", data_dir + "h6/channel_2.dat", data_dir + "h6/channel_8.dat")
        if type !="Overlap":
            aggregate6, individual6 = data.convert_to_tensor(aggregate6, individual6, REFRIGERATOR_WINDOW_SIZE)
        else:
            aggregate6, individual6 = data.convert_to_tensor_overlap(aggregate6, individual6, REFRIGERATOR_WINDOW_SIZE)

        self.aggregate = torch.cat((aggregate1, aggregate3, aggregate2, aggregate6), 0 )
        self.individual = torch.cat((individual1,individual3, individual2, individual6), 0)
        # self.mean = aggregate.numpy().mean()
        # self.sd = aggregate.numpy().sd()

    def init_transformation(self, transform):
        if not self.transform:
            self.transform = transform
        else:
            print("Transformations are already predefined and you cannot initialize another transformations.")
    def __len__(self):
        return len(self.aggregate)

    def __getitem__(self, index):
        """Geting one item from a dataset in format: [input(aggregate), desired_output(individual)]"""
        aggregate = self.aggregate[index]
        individual = self.individual[index]
        if self.transform:
            sample = {}
            sample["Aggregate"] = aggregate
            sample["Individual"] = individual
            aggregate, individual = self.transform(sample)
        return aggregate, individual

    def get_mean(self):
        return self.aggregate.numpy().mean()
    def get_sd(self):
        return self.aggregate.numpy().std()

class MicrowaveREDDDataSet(torch_data.Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        #Reading two mains and microwave data from house1 (Nascimento used this data for training)
        aggregate1, individual1 = data.downsampled_channels(data_dir + "h1/channel_1.dat", data_dir + "h1/channel_2.dat", data_dir + "h1/channel_11.dat")
        #Conversion of data to 2D torch Tensors
        aggregate1,individual1 = data.convert_to_tensor(aggregate1, individual1, MICROWAVE_WINDOW_SIZE)

        #Reading two mains and microwave data from house2 (Nascimento used this data for training)
        aggregate2, individual2 = data.downsampled_channels(data_dir + "h2/channel_1.dat", data_dir + "h2/channel_2.dat", data_dir + "h2/channel_6.dat")
        aggregate2,individual2 = data.convert_to_tensor(aggregate2, individual2, MICROWAVE_WINDOW_SIZE)

        #Reading two mains and microwave data from house3 (Nascimento used this data for testing)
        aggregate3, individual3 = data.downsampled_channels(data_dir + "h3/channel_1.dat", data_dir + "h3/channel_2.dat", data_dir + "h3/channel_16.dat")
        aggregate3,individual3 = data.convert_to_tensor(aggregate3, individual3, MICROWAVE_WINDOW_SIZE)

        self.aggregate = torch.cat((aggregate1, aggregate2, aggregate3), 0)
        self.individual = torch.cat((individual1, individual2, individual3), 0)

    def __len__(self):
        return len(self.aggregate)

    def __getitem__(self, index):
        """Geting one item from a dataset in format: [input, desired_output]"""
        if index >= 0 and index < self.__len__():
            return self.aggregate[index], self.individual[index]
        else:
            return None

class DishwasherREDDDataSet(torch_data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        #Reading two mains and dishwasher data from house1 (Nascimento used this data for training)
        aggregate1, individual1 = data.downsampled_channels(data_dir + "h1/channel_1.dat", data_dir + "h1/channel_2.dat", data_dir + "h1/channel_6.dat")
        #Conversion of data to 2D torch Tensors
        aggregate1,individual1 = data.convert_to_tensor(aggregate1, individual1, DISHWASHER_WINDOW_SIZE)

        #Reading two mains and dishwasher data from house2 (Nascimento used this data for testing)
        aggregate2, individual2 = data.downsampled_channels(data_dir + "h2/channel_1.dat", data_dir + "h2/channel_2.dat", data_dir + "h2/channel_10.dat")
        aggregate2,individual2 = data.convert_to_tensor(aggregate2, individual2, DISHWASHER_WINDOW_SIZE)

        #Reading two mains and dishwasher data from house3 (Nascimento used this data for training)
        aggregate3, individual3 = data.downsampled_channels(data_dir + "h3/channel_1.dat", data_dir + "h3/channel_2.dat", data_dir + "h3/channel_9.dat")
        aggregate3,individual3 = data.convert_to_tensor(aggregate3, individual3, DISHWASHER_WINDOW_SIZE)

        #Reading two mains and dishwasher data from house4 (Nascimento used this data for testing)
        aggregate4, individual4 = data.downsampled_channels(data_dir + "h4/channel_1.dat", data_dir + "h4/channel_2.dat", data_dir + "h4/channel_15.dat")
        aggregate4,individual4 = data.convert_to_tensor(aggregate4, individual4, DISHWASHER_WINDOW_SIZE)

        self.aggregate = torch.cat((aggregate1, aggregate2, aggregate3, aggregate4), 0)
        self.individual = torch.cat((individual1, individual2, individual3, individual4), 0)

    def get_mean(self):
        return self.aggregate.numpy().mean()

    def get_sd(self):
        return self.aggregate.numpy().std()

    def init_transformation(self, transform):
        if not self.transform:
            self.transform = transform
        else:
            print("Transformations are already predefined and you cannot initialize another transformations.")

    def __len__(self):
        return len(self.aggregate)

    def __getitem__(self, index):
        """Geting one item from a dataset in format: [input(aggregate), desired_output(individual)]"""
        aggregate = self.aggregate[index]
        individual = self.individual[index]
        if self.transform:
            sample = {}
            sample["Aggregate"] = aggregate
            sample["Individual"] = individual
            aggregate, individual = self.transform(sample)
        return aggregate, individual

class DishwasherFridgeREDDDataset(torch_data.Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        #Loading data from house h1
        aggregate1, fridge1, dishwasher1 = data.downsampled_channels_combined(data_dir + "h1/channel_1.dat", data_dir + "h1/channel_2.dat", data_dir + "h1/channel_5.dat", data_dir + "h1/channel_6.dat")
        aggregate1, fridge1, dishwasher1 = data.convert_to_tensor_combined(aggregate1, fridge1, dishwasher1, REFRIGERATOR_DISHWASHER_WINDOW_SIZE)
        #Loading data from house h2
        aggregate2, fridge2, dishwasher2 = data.downsampled_channels_combined(data_dir + "h2/channel_1.dat", data_dir + "h2/channel_2.dat", data_dir + "h2/channel_9.dat", data_dir + "h1/channel_10.dat")
        aggregate2, fridge2, dishwasher2 = data.convert_to_tensor_combined(aggregate2, fridge2, dishwasher2, REFRIGERATOR_DISHWASHER_WINDOW_SIZE)
        #Loading data from house h3
        aggregate3, fridge3, dishwasher3 = data.downsampled_channels_combined(data_dir + "h3/channel_1.dat", data_dir + "h3/channel_2.dat", data_dir + "h3/channel_7.dat", data_dir + "h3/channel_9.dat")
        aggregate3, fridge3, dishwasher3 = data.convert_to_tensor_combined(aggregate3, fridge3, dishwasher3, REFRIGERATOR_DISHWASHER_WINDOW_SIZE)
        #Convert to tensors

        self.aggregate = torch.cat((aggregate1, aggregate2, aggregate3), 0)
        self.refrigerator = torch.cat((fridge1, fridge2, fridge3), 0)
        self.dishwasher = torch.cat((dishwasher1, dishwasher2, dishwasher3), 0)
    def get_mean(self):
        return self.aggregate.numpy().mean()

    def get_sd(self):
        return self.aggregate.numpy().std()

    def __len__(self):
        return len(self.aggregate)

    def init_transformation(self, transform):
        if not self.transform:
            self.transform = transform
        else:
            print("Transformations are already predefined and you cannot initialize another transformations.")

    def __getitem__(self, index):
        """Geting one item from a dataset in format: [input(aggregate), desired_output(individual)]"""
        aggregate = self.aggregate[index]
        refrigerator = self.refrigerator[index]
        dishwasher = self.dishwasher[index]
        if self.transform:
            sample = {}
            sample["Aggregate"] = aggregate
            sample["Individual1"] = refrigerator
            sample["Individual2"] = dishwasher
            aggregate, refrigerator, dishwasher = self.transform(sample)
        return aggregate, refrigerator, dishwasher

class REDDCleanDataset(torch_data.Dataset):
    def __init__(self, data_dir, transform=None, appliance='Refrigerator', window_size='2401'):
        self.data_dir = data_dir
        self.appliance = appliance
        self.window_size = window_size
        self.transform = transform
        self.data = data.generate_clean_data(self.data_dir, self.appliance, self.window_size)

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
