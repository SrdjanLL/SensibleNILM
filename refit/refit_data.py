import pandas as pd
import numpy as np
import torch
from refit_parameters import *
import matplotlib.pyplot as plt
import random

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

def create_overlap_windows(agg, iam, window_size, stride=100):
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

def show_example(agg, iam, window_size, title=None):
    plt.plot([i * 8 for i in range(window_size)], agg.numpy().flatten(), color='C1', label='Aggregate power consumption')
    plt.plot([i * 8 for i in range(window_size)], iam.numpy().flatten(), color='C2', label='Individual power consumption')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.legend()
    plt.show()

def show_output(aggregate, individual, output, label, window_size):
    if torch.cuda.is_available():
        aggregate= aggregate.cpu()
        individual = individual.cpu()
        output = output.cpu()

    plt.plot(range(1, 8 * window_size, 8), aggregate.numpy(), 'C1', label='Aggregate usage')
    plt.plot(range(1, 8 * window_size, 8), individual.numpy(), 'C2', label='Individual usage')
    plt.plot(range(1, 8 * window_size, 8), [label for i in range(window_size)], 'C3', label='Real average usage')
    plt.plot(range(1, 8 * window_size, 8), [output.numpy() for i in range(window_size)], 'C4', label='Predicted average usage')
    plt.tight_layout()
    plt.legend()
    plt.show()

def house_subset(file_name, file_length, channel, window_size, readfrom=0, count=1, stride=1):
    """Create data-windows subset from a house in refit dataset. Good for batch reading so that not all data is loaded into memory at the same time.
        Args:
            file_name: relative or absolute path to the cleaned refit house csv file from where data will be read
            file_length: number of samples in that house (for prevention of index out of boundary exceptions)
            channel: string name of the column where targeted appliances signals are stored
            window_size: integer representing number of samples contained in one window
            readfrom: integer which contains information from which sample to read withing csv file
            count: integer number of sliding windows expected to be returned by method
            stride: integer number representing standard step which is conducted after each window is created

        Return:
                - integer which represents current indicator within refit house csv file after batch reading is Finished
                - python array of 2 elements where first is 2D numpy array of aggregate data and second one is 2D numpy array of tageted appliance's data. Each two rows of those arrays respond to one example. """
    df = pd.read_csv(file_name, skiprows=range(1, readfrom), nrows=window_size + (count-1) * stride, usecols=['Aggregate', channel])
    agg = np.array(df['Aggregate'])
    iam = np.array(df[channel])
    start_position = 0
    agg_set =  []
    iam_set = []
    for i in range(count):
        agg_buff, iam_buff = agg[start_position:start_position + window_size], iam[start_position:start_position + window_size]
        if len(agg_buff) == window_size and readfrom+window_size < file_length:
            agg_set.append(agg_buff)
            iam_set.append(iam_buff)
            start_position += stride
            readfrom += stride

    zipper = list(zip(agg_set, iam_set))
    random.shuffle(zipper)
    agg_set, iam_set = zip(*zipper)
    data = [np.array(agg_set), np.array(iam_set)]
    return start_position + stride, data

# def plot_test_set(net, test_set, window_size):
#     """Method for ploting network's predictions"""
#     testloader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=2)
#     for i in range(len(test_set)):
#         aggregate, labels = test_set[i]
#         label = labels.mean()
#         inputs = aggregate
#         inputs = inputs.view(-1, 1, window_size)
#         if torch.cuda.is_available():
#             inputs, labels = inputs.cuda(), labels.cuda()
#         inputs = Variable(inputs.float())
#         outputs = net(inputs)
#         #print('Outputs:', outputs)
#         show_output(aggregate=aggregate, individual=labels, output=outputs.data[0], label=label, window_size=params.REFRIGERATOR_WINDOW_SIZE)
