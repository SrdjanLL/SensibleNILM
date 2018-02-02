import sys
sys.path.insert(0, '..')
import numpy as np
import torch
import redd_parameters as params
import redd_data as data
import dataset
from matplotlib import pyplot as plt
from transformations import Normalize
import torch.nn as nn
import torch.optim as optim
from models import *
from torch.autograd import Variable
import torchvision
import scores
import train

#Initialization of dataset
#training_set, test_set = data.generate_clean_data(data_dir="../data/REDD/", appliance='Dishwasher', window_size=params.DISHWASHER_WINDOW_SIZE, proportion=[0, 0], threshold=10, test=True, test_on='All', stride=1) #stride=1

#training_set = dataset.REDDDataset(data=training_set)
#test_set = dataset.REDDDataset(data=test_set)
#mean, std = training_set.get_mean_and_std()
mean = 395.890660893
std = 615.928606564
# mean = 200
# std = 450
#training_set.init_transformation(torchvision.transforms.Compose([Normalize(mean=mean, sd=std)]))
#del training_set, test_set
training_set, test_set = data.generate_clean_data(data_dir="../data/REDD/", appliance='Dishwasher', window_size=params.DISHWASHER_WINDOW_SIZE, proportion=[3, 0], threshold=10, test=True, test_on='h1', stride=100) #stride=1
mean = 395.890660893
std = 615.928606564
# test_set = data.read_from_home('../data/REDD/', 'h2', appliance='Dishwasher', window_size=params.DISHWASHER_WINDOW_SIZE)
test_set = dataset.REDDDataset(data=test_set)
test_set.init_transformation(torchvision.transforms.Compose([Normalize(mean=mean, sd=std)]))
print('Size of test set: ', len(test_set))
# del training_set
net = ConvDishNILM()
try:
    net = torch.load('models/dishwasher_lucky_shot_house1.pt')
except FileNotFoundError:
    print('There is no pretrained model')

if torch.cuda.is_available():
    net = net.cuda()
print('Testing.')
print("Trained network's results: ")
net_scores = train.test_network(net, test_set, params.DISHWASHER_WINDOW_SIZE, std, mean)
for i in range(len(test_set)):
    aggregate, labels = test_set[i]
    label = labels.mean()
    inputs = aggregate
    inputs = inputs.view(-1, 1, params.DISHWASHER_WINDOW_SIZE)
    if torch.cuda.is_available():
        inputs, labels = inputs.cuda(), labels.cuda()
    inputs = Variable(inputs.float())
    outputs = net(inputs)
    #print('Outputs:', outputs)
    data.show_output(aggregate=aggregate, individual=labels, output=outputs.data[0][0], label=label, window_size=params.DISHWASHER_WINDOW_SIZE)
