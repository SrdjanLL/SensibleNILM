import pandas as pd
from os import walk
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

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
    """Returns dictionary of homes with keys "h1", "h2" etc, where each channel is a panda Dataframe
        Args: path - string containing path to where REDD home folders are located.
        Return: dictionary in format:
                                         { h1: {
                                                channel_1.dat: {
                                                    values
                                                },
                                                channel_2.dat: {
                                                    values
                                                }. . .
                                            },
                                            h2: {

                                            }. . .
                                         }
    """
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
    """Summing two mains channels of a house from dataset so that real aggregate power usage is returned.
        Args:
                channel_1 : path to channel_1.dat in a house(n)
                channel_2 : path to channel_2.dat in a house(n)

        return:
                pandas.Datataframe containing sum of Individual_usage from two channels labeled as mains
    """
    #reading of two mains channels
    aggregate1 = read_channel(channel_1)
    aggregate2 = read_channel(channel_2)
    sum_channels = np.array(aggregate1["Individual_usage"]) + np.array(aggregate2["Individual_usage"])
    aggregate1["Individual_usage"] = sum_channels
    return aggregate1

def downsampled_channels(_aggregate1, _aggregate2, _individual_appliance, read=True):
    """This method is downsampling aggregate data from 1s data to ~4s data as it was sampled for each individual appliance in REDD data
        Args:
                     _aggregate1 : pandas.Dataframe containing channel_1.dat in a house h(n)
                     _aggregate2 : pandas.Dataframe containing channel_2.dat in a house h(n)
                     _individual_appliance : pandas.Dataframe containing channel of individual appliance in a home from where downsampling is done

        return:
                    [pandas.Dataframe] aggregate, individual_appliance - so it can be converted in a torch tensor after that (and possibly some other transformations)
    """

    aggregate = join_aggregate_data(_aggregate1, _aggregate2)
    individual_appliance = read_channel(_individual_appliance)
    #Dropping aggregate samples which have unix time not contained in time column of individual_appliance
    individual_appliance = individual_appliance[individual_appliance["Time"].isin(aggregate["Time"])]
    aggregate = aggregate[aggregate["Time"].isin(individual_appliance["Time"])]
    #print("Aggregate data length: ", len(aggregate), ", Individual appliance length: ", len(individual_appliance))
    return aggregate, individual_appliance

def generate_clean_data(data_dir, appliance, window_size, threshold, proportion=[1,1],test=False, test_on='All'):
    activation_proportion = proportion[0]
    non_activation_proportion = proportion[1]
    aggregate_channels = []
    individual_channels = []
    aggregate_channels_test = []
    individual_channels_test = []
    activations = []
    non_activations = []
    channels = []
    if appliance == 'Refrigerator':
        channels = [['h1', 'channel_5.dat'], ['h2', 'channel_9.dat'], ['h3', 'channel_7.dat'], ['h6', 'channel_8.dat']] #channels and houses from which fridge data is read
    elif appliance == 'Dishwasher':
        channels = [['h1', 'channel_6.dat'], ['h3', 'channel_9.dat'], ['h4', 'channel_15.dat'], ['h2', 'channel_10.dat']] #channels and houses from which dishwasher data is read
    elif appliance == 'Microwave':
        channels = [['h2', 'channel_6.dat'],['h3', 'channel_16.dat'], ['h1', 'channel_11.dat']]

    for house, channel in channels:
        channel1 = data_dir + house + '/channel_1.dat'
        channel2 = data_dir + house + '/channel_2.dat'
        channel3 = data_dir + house + '/' + channel
        aggregate, iam = downsampled_channels(channel1, channel2, channel3) #downsampling aggregate data so that it contains all timestamps as iam data
        aggregate, iam = np.array(aggregate['Individual_usage']), np.array(iam['Individual_usage']) #converting downsampled pandas.Dataframes to numpy arrays
        if test:
            split = round(len(aggregate) * 0.8)
            aggregate_test = aggregate[split:]
            iam_test = iam[split:]
            if test_on == 'All':
                aggregate_channels_test.append(aggregate_test)
                individual_channels_test.append(iam_test)
                aggregate = aggregate[:split]
                iam = iam[:split]
            elif test_on == house:
                aggregate_channels_test.append(aggregate_test)
                individual_channels_test.append(iam_test)
                aggregate = aggregate[:split]
                iam = iam[:split]
        aggregate_channels.append(aggregate) #appending the aggregate to aggregate list and iam to iam list so that their indices match
        individual_channels.append(iam)
    for channel in individual_channels: #iterating through frigde power usage in each house
        activations_for_house = [] #buffer list to fill all activations detected in iam
        non_activations_for_house = []
        non_activation_samples = 0
        for i in range(len(channel)):
            start = 0
            end = 0
            if channel[i] > threshold: #if power is above threshold power that may possibly be an activation
                if non_activation_samples > window_size:
                    non_activations_for_house.append([i - non_activation_samples, i-1])
                non_activation_samples = 0
                start = i
                while channel[i] > threshold and i < len(channel) - 1:
                    i += 1 #increasing index indicator until it reaches the value below threshold
                end = i
                activation = [start, end]
                activations_for_house.append(activation) #appending activation start and end time to buffer of activations for house
            else:
                non_activation_samples +=1
        activations.append(activations_for_house) #appending whole bufer to list of activations of specific appliance in all houses used for loading activations
        non_activations.append(non_activations_for_house)

    agg, iam = [], []
    for i in range(len(aggregate_channels)):
        #iterating through aggregate data of each house
        print('Number of activations in this channel: ', len(activations[i]))
        print('Number of non-activations in this channel: ', len(non_activations[i]))
        agg_windows, iam_windows = create_overlap_windows(aggregate_channels[i], individual_channels[i], window_size, stride=5)
        agg.extend(agg_windows)
        iam.extend(iam_windows)
        for start, end in activations[i]:
            #then iterating through activation positions in specified house [i]
            for j in range(activation_proportion):
                #randomly generate windows #n times with one activation
                activation_size = end - start
                #randomly positioning activation in window

                start_aggregate = start - random.randint(0, window_size - activation_size)
                #if start_aggregate + window_size < len(aggregate_channels[i]):
                agg_buff, iam_buff = aggregate_channels[i][start_aggregate: start_aggregate + window_size], individual_channels[i][start_aggregate: start_aggregate + window_size]
                agg_buff, iam_buff = np.copy(agg_buff), np.copy(iam_buff)
                agg.append(agg_buff)
                iam.append(iam_buff)
        for start, end in non_activations[i]:
            for j in range(non_activation_proportion):
                window_start = random.randint(start, end - window_size)
                agg_buff, iam_buff = aggregate_channels[i][window_start: window_start + window_size], individual_channels[i][window_start: window_start + window_size]
                agg_buff, iam_buff = np.copy(agg_buff), np.copy(iam_buff)
                agg.append(agg_buff)
                iam.append(iam_buff)
    zipper = list(zip(agg, iam))
    random.shuffle(zipper)
    agg, iam = zip(*zipper)
    agg, iam = np.array(agg), np.array(iam)
    dataset = [agg, iam]

    #Creating test set if test==True
    agg_test = []
    iam_test = []
    isFirst = True
    if test:
        for i in range(len(aggregate_channels_test)):
            agg_buff_test, iam_buff_test = create_windows(aggregate_channels_test[i], individual_channels_test[i], window_size=window_size)
            if isFirst:
                agg_test = agg_buff_test
                iam_test = iam_buff_test
                isFirst = False
            else:
                print(agg_test)
                print(agg_buff_test)
                agg_test = np.concatenate((agg_test, agg_buff_test), axis=0)
                iam_test = np.concatenate((iam_test, iam_buff_test), axis=0)
        testset = [agg_test, iam_test]
        return dataset, testset

    return dataset

def create_windows(agg, iam, window_size):
    #Creating padding so the last row of numpy.array can fit to window size for specified appliance
    zeros_padding = window_size - len(agg)%window_size
    #Appending zeros padding to the end of numpy arrays of both individual and aggregate
    agg = np.append(agg, np.zeros(zeros_padding))
    iam = np.append(iam, np.zeros(zeros_padding))
    agg = np.reshape(agg, (-1, window_size))
    iam = np.reshape(iam, (-1, window_size))
    agg = agg[:len(agg)-2]
    iam = iam[:len(iam)-2]
    return agg, iam

def create_overlap_windows(agg, iam, window_size, stride=10):
    position = 0
    agg_windows = []
    iam_windows = []
    while position < len(agg) - window_size -1:
        agg_buffer = agg[position: position + window_size]
        iam_buffer = iam[position: position + window_size]
        agg_windows.append(agg_buffer)
        iam_windows.append(iam_buffer)
        position += stride
    return agg_windows, iam_windows

def show_example(aggregate, individual, window_size, title=None):
    if title:
        plt.title(title)
    plt.plot(range(1, 4 * window_size, 4), aggregate.numpy(), 'C1', label='Aggregate power consumption')
    plt.plot(range(1, 4 * window_size, 4), individual.numpy(), 'C2', label=title + ' power consumption')
    plt.tight_layout()
    plt.legend()
    plt.show()

def show_output(aggregate, individual, output, label, window_size):
    if torch.cuda.is_available():
        aggregate= aggregate.cpu()
        individual = individual.cpu()
    plt.plot(range(1, 4 * window_size, 4), aggregate.numpy(), 'C1', label='Aggregate usage')
    plt.plot(range(1, 4 * window_size, 4), individual.numpy(), 'C2', label='Individual usage')
    plt.plot(range(1, 4 * window_size, 4), [label for i in range(window_size)], 'C3', label='Real average usage')
    plt.plot(range(1, 4 * window_size, 4), [output for i in range(window_size)], 'C4', label='Predicted average usage')
    plt.tight_layout()
    plt.legend()
    plt.show()
