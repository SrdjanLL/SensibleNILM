import pandas as pd
from os import walk
import numpy as np
import torch
def read_channel(filename):
    """Method to read home channel data from .dat file into panda dataframe
        Args:
                filename: path to a specific channel_(m) from house(n)

        return:
                [pandas.Dataframe] of a signle channel_(m) from house(n)
    """
    channel_to_read = pd.read_csv(filename, names=["Time", "Individual_usage"], delim_whitespace=True)

    return channel_to_read

def read_all_homes(path): #Point to REDD data directory with homes directories structured as h1, h2, h3, etc.
    """Returns dictionary of homes with keys "h1", "h2" etc, where each channel is a panda dataframe"""
    homes = {}
    mypath = path
    for (dirpath, dirnames, filenames) in walk(mypath):
        for dirname in dirnames:
            homes[dirname] = {}
            home = homes[dirname]
            for (dirpath, dirnames, filenames) in walk(mypath + dirname):
                print("Entered: ", mypath + dirname)
                for filename in filenames:
                    filepath = mypath + dirname + "/" + filename
                    print("Reading: ", filepath)
                    home[filename] = read_channel(filepath)
    return homes

def join_aggregate_data(channel_1, channel_2):
    """Summing between two mains of a house from dataset.
        Args:
                channel_1 : path to channel_1.dat in a house(n)
                channel_2 : path to channel_2.dat in a house(n)

        return:
                pandas.Datataframe containing sum of Individual_usage from two channels labeled as mains
    """
    aggregate1 = read_channel(channel_1)
    aggregate2 = read_channel(channel_2)
    sum_channels = np.array(aggregate1["Individual_usage"]) + np.array(aggregate2["Individual_usage"])
    aggregate1["Individual_usage"] = sum_channels
    return aggregate1

def downsampled_channels(_aggregate1, _aggregate2, _individual_appliance):
    """This method is downsampling aggregate data from 1s data to ~4s data as it was sampled for each individual appliance in REDD data
        Args:
                     _aggregate1 : path to channel_1.dat in a house h(n)
                     _aggregate2 : path to channel_2.dat in a house h(n)
                     _individual_appliance : path to a channel of individual appliance in a home from where downsampling is done

        return:
                    individual_appliance, aggregate - so it can be converted in a torch tensor after that (and possibly some other transformations)
    """
    aggregate = join_aggregate_data(_aggregate1, _aggregate2)
    individual_appliance = read_channel(_individual_appliance)

    #Dropping aggregate samples which have unix time not contained in time column of individual_appliance
    individual_appliance = individual_appliance[individual_appliance["Time"].isin(aggregate["Time"])]
    aggregate = aggregate[aggregate["Time"].isin(individual_appliance["Time"])]
    #print("Aggregate data length: ", len(aggregate), ", Individual appliance length: ", len(individual_appliance))
    return individual_appliance, aggregate


def convert_to_tensor(aggregate, individual, window_size):
    """Method for converting aggregate and individual appliance monitoring data to two torch 2D tensors of the same size so aggregate data is input and
        individual data is desired output.
        Args:
                aggregate: pandas.Dataframe containing concatenated and downsampled channel data
                individual: pandas.Dataframe containing channel data for individual appliance
                NOTE: the lengths of aggregate and individual data must be the same.

        return aggregate, individual : Two torch Tensors where each row is of specified window size.
    """

    #Converting to numpy array containing only usage (dropping Time column completely).
    aggregate = np.array(aggregate["Individual_usage"])
    individual = np.array(individual["Individual_usage"])

    #Creating padding so the last row of torch.Tensor can fit to window size of specifiedAppliance
    zeros_padding = window_size - len(aggregate)%window_size

    #Appending zeros padding to the end of numpy arrays of both individual and aggregate
    aggregate = np.append(aggregate, np.zeros(zeros_padding))
    individual = np.append(individual, np.zeros(zeros_padding))

    #Conversion to 1D torch.Tensor from numpy
    aggregate = torch.from_numpy(aggregate)
    individual = torch.from_numpy(individual)

    #Reshaping from 1D to 2d toch.Tensor
    aggregate = aggregate.view(-1, window_size)
    individual = individual.view(-1, window_size)

    return aggregate, individual

#myHomes = read_all_homes("data/REDD/")
#print(downsampled_channels("data/REDD/h1/channel_1.dat", "data/REDD/h1/channel_2.dat", "data/REDD/h1/channel_3.dat"))
