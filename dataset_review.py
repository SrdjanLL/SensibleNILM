import numpy as np
import torch
from parameters import *
from data import *
from dataset import *
from matplotlib import pyplot as plt
from transformations import Normalize
import torch.nn as nn
import torch.optim as optim
from models import *
from torch.autograd import Variable
import torchvision

# dishwasher_train_set = REDDCleanDataset(data_dir="data/REDD/", appliance='Dishwasher', window_size=DISHWASHER_WINDOW_SIZE, proportion=[3, 900], threshold=10)
# dishwasher_test_set = REDDCleanDataset(data_dir="data/REDD/", appliance='Dishwasher', window_size=DISHWASHER_WINDOW_SIZE, test=True)
train_set, test_set = generate_clean_data2(data_dir="data/REDD/", appliance='Dishwasher', window_size=DISHWASHER_WINDOW_SIZE, proportion=[2, 1000], threshold=10, test=True, test_on='h4')
dishwasher_test_set = REDDDataset(data=test_set)
#print('Training set length: ', len(train_set))
print('Test set length: ', len(dishwasher_test_set))

for i in range(len(dishwasher_test_set)):
    print('Example ', (i+1), ': ')
    agg, iam = dishwasher_test_set[i]
    show_example(agg, iam, DISHWASHER_WINDOW_SIZE, title='Dishwasher')
