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

def train_network(net, training_set, window_size, epochs, batch_size): #put all parameters
    """Method which trains NILM neural networks for a given number of epochs.
        Args:
                net: object which represents the model. Object must be derived from a class that inherits pytorch's torch.nn.Module class. Could be newly instantiated or pretrained.
                training_set: one of the Datasets from dataset module in the parent directory.
                window_size: number of samples in time window that represent one training example
                epochs: number of training epochs (ex. range(5))
                batch_size: number of examples inside of a single batch

        Return:
                net: trained network with previously specified training parameters."""
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, num_workers=8)

    #Defining Mean Squared Error (MSE) as loss function
    criterion = nn.MSELoss()

    #Optimization method: Stohastic Gradient Descent with momentum of 0.9 and learning rate of 0.001
    optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)

    for epoch in epochs:
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            #Get the inputs
            if i == len(train_loader) - 1:
                continue
            inputs, labels = data
            #labels would represent iam full signature within one window
            #label is the average consumption within iam's window and it represents the target of the network
            label = labels.mean(dim=1)
            inputs = inputs.view(-1, 1, window_size)

            #Switching to torch.cuda.Tensor for improved training performance
            if torch.cuda.is_available():
                inputs, labels, label = inputs.cuda(), labels.cuda(), label.cuda()

            #Wrapping inputs and label into torch.autograd.Variable
            inputs, label = Variable(inputs.float()), Variable(label.float()) #Float because network works with float numbers

            #Zero the parameter gradients
            optimizer.zero_grad()

            #forward + backward + optimizer
            outputs = net(inputs)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            #Print statistics
            running_loss += loss.data[0]
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
    return net


def test_network(net, test_set, window_size, std, mean):
    """Method which tests NILM neural networks on a given test set (which the network usually doesn't see during training).
        Args:
                net: object which represents the model. Object must be derived from a class that inherits pytorch's torch.nn.Module class.
                training_set: object of Dataset class from dataset module in the parent directory.
                window_size: number of samples in time window that represent one training example
                std, mean: values used for denormalization of the test set for metrics requiring values in watts.

        Return:
                net_scores: test scores which represent networks generalization performance on a test set."""
    net = net.eval()
    net_scores = scores.get_scores(net, test_set, 1, window_size, std, mean)
    return net_scores
