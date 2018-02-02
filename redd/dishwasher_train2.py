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

# As test_on you for dishwasher you can put 'h1', 'h2', 'h3', 'h4', 'All', also best thing is that stride=1 (default), but careful with RAM
# Code for generate_clean_data is way to messy for you to worry about it, just call it :)
training_set, test_set = data.generate_clean_data(data_dir="../data/REDD/", appliance='Dishwasher', window_size=params.DISHWASHER_WINDOW_SIZE, proportion=[3, 0], threshold=10, test=True, test_on='h1',stride=2)

training_set = dataset.REDDDataset(data=training_set)
test_set = dataset.REDDDataset(data=test_set)

# These are exctracted from aggregate data of all the houses so I use them as constants for every dataset's normalization for now.
mean = 395.890660893
std = 615.928606564
# Dataset normalization [must do]
training_set.init_transformation(torchvision.transforms.Compose([Normalize(mean=mean, sd=std)]))
test_set.init_transformation(torchvision.transforms.Compose([Normalize(mean=mean, sd=std)]))

# This in initialization of network from models.py file, put your network model over there and call it the same way like this
net = ConvDishNILM()
try:
    net = torch.load('models/dishwasher_model.pt')
except FileNotFoundError:
    print('There is no pretrained model')

if torch.cuda.is_available():
    net = net.cuda() #switch to cuda
print('Start of training.')
for i in range(1):
    # Send any range you want for a number of epochs. Net is the model from models.py file, training_set should be dataset.REDDDataset object, window size as usual and last parameter is the batch size
    net = train.train_network(net, training_set, params.DISHWASHER_WINDOW_SIZE, range(1), 32)
    print('Testing.')
    print("Trained network's results: ")
    # Same as train.train_newtork, but you have to send standard deviation and mean as parameters because denormalization is done for score computing
    net_scores = train.test_network(net, test_set, params.DISHWASHER_WINDOW_SIZE, std, mean)
    torch.save(net, 'models/dishwasher_model.pt')
