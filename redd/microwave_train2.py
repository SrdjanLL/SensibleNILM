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
training_set, test_set = data.generate_clean_data(data_dir="../data/REDD/", appliance='Microwave', window_size=params.MICROWAVE_WINDOW_SIZE, proportion=[15, 0], threshold=200, test=True, test_on='All') #stride=1

training_set = dataset.REDDDataset(data=training_set)
test_set = dataset.REDDDataset(data=test_set)
mean = 395.890660893
std = 615.928606564
training_set.init_transformation(torchvision.transforms.Compose([Normalize(mean=mean, sd=std)]))
test_set.init_transformation(torchvision.transforms.Compose([Normalize(mean=mean, sd=std)]))

net = ConvMicroNILM()
try:
    net = torch.load('models/microwave_model.pt')
except FileNotFoundError:
    print('There is no pretrained model')


if torch.cuda.is_available():
    net = net.cuda()
print('Start of training.')
net = train.train_network(net, training_set, params.MICROWAVE_WINDOW_SIZE, range(1,5), 32)
print('Testing.')
print("Trained network's results: ")
net_scores = train.test_network(net, test_set, params.MICROWAVE_WINDOW_SIZE, std, mean)
torch.save(net.cpu(), 'models/microwave_model.pt')

del training_set, test_set, net
