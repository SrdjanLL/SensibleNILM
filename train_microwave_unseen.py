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

#Initialization of dataset
microwave_train_set = REDDCleanDataset(data_dir="data/REDD/", appliance='Microwave', window_size=MICROWAVE_WINDOW_SIZE)
microwave_test_set = REDDCleanDataset(data_dir="data/REDD/", appliance='Microwave', window_size=MICROWAVE_WINDOW_SIZE, test=True)
#Getting mean and standard deviation so Normalization can be performed
mean, std = microwave_train_set.get_mean_and_std()
microwave_train_set.init_transformation(torchvision.transforms.Compose([Normalize(mean=mean, sd=std)]))
print('Dataset size: ', len(microwave_train_set))

#Splitting dataset into training and test set
train_examples_count = round(len(microwave_train_set) * 0.8)
microwave_trainloader = torch.utils.data.DataLoader(microwave_train_set, batch_size=1, num_workers=2)
microwave_testloader = torch.utils.data.DataLoader(microwave_test_set, batch_size=1, num_workers=2)

#Initialization of neural network
net = ConvMicroNILM2()
if torch.cuda.is_available():
    net = net.cuda()
criterion = nn.MSELoss()
optimimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)
print("Start of training: ")
for epoch in range(30):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(microwave_trainloader, 0):
        #Get the inputs
        inputs, labels = data
        label = torch.Tensor([labels.mean()]).float()
        # if label[0] < (10-mean)/sd or labels.numpy().std() < 0.05:
        #     label[0] = (0 - mean)/ sd
        inputs = inputs.view(1, -1, MICROWAVE_WINDOW_SIZE)
        #Wrap them in variables
        if torch.cuda.is_available():
            inputs, labels, label = inputs.cuda(), labels.cuda(), label.cuda()

        inputs = Variable(inputs.float())

        #Zero the parameter gradients
        optimimizer.zero_grad()

        #forward + backward + optimimizer
        outputs = net(inputs)
        loss = criterion(outputs, Variable(label))
        loss.backward()
        optimimizer.step()

        #Print statistics
        running_loss += loss.data[0]
        if i % 1000 == 999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0
    net.eval()
    score_mae = None #Mean absolute error
    score_ptecc = 0 #Proportion of total energy correctly classified
    denominator = 0 #denominator for score_ptecc metric and MNE
    score_mne = 0 #Mean normalised error
    mne_estimated = 0
    mne_label = 0
    count = 0
    score_mse = 0 #Mean Squared Error
    for i, data in enumerate(microwave_testloader, 0):
        count += 1
        inputs, labels = data
        label = torch.Tensor([labels.mean()]).float()
        inputs = inputs.view(1, -1, MICROWAVE_WINDOW_SIZE)
        if torch.cuda.is_available():
            inputs, labels, label = inputs.cuda(), labels.cuda(), label.cuda()

        inputs = Variable(inputs.float())
        outputs = net(inputs)
        output = outputs.data * std + mean
        label = label * std + mean
        err = sum(torch.abs(output-label))

        #mse
        mse_err = sum(pow(torch.abs(output - label),2))
        score_mse += mse_err
        #ptec
        score_ptecc += torch.abs(output - label)
        denominator += label

        #mne
        mne_estimated += output
        mne_label += label
        if i==0 :
            score_mae = err
        else :
            score_mae = torch.cat((score_mae, err), 0)

    score_ptecc /= 2 * denominator
    score_ptecc = (1 - score_ptecc) * 100
    score_mae = score_mae.mean()
    score_mne = torch.abs(mne_estimated - mne_label)/denominator
    score_mse /= count
    print('---------------------------')
    print('Proportion of energy correctly classified: ', score_ptecc[0][0], '%')
    print('Mean absolute error: ', score_mae)
    print('Mean normalised error: ', score_mne[0][0])
    print('Mean Squared Error: ', score_mse[0])
    print('Root Mean Squared Error: ', torch.sqrt(score_mse)[0])
    print('--------------------------')
print('Finished Training')

net.eval()
score_mae = None #Mean absolute error
score_ptecc = 0 #Proportion of total energy correctly classified
denominator = 0 #denominator for score_ptecc metric and MNE
score_mne = 0 #Mean normalised error
mne_estimated = 0
mne_label = 0
count = 0
score_mse = 0 #Mean Squared Error
for i, data in enumerate(microwave_testloader, 0):
    count += 1
    inputs, labels = data
    label = torch.Tensor([labels.mean()]).float()
    inputs = inputs.view(1, -1, MICROWAVE_WINDOW_SIZE)
    if torch.cuda.is_available():
        inputs, labels, label = inputs.cuda(), labels.cuda(), label.cuda()

    inputs = Variable(inputs.float())
    outputs = net(inputs)
    output = outputs.data * std + mean
    label = label * std + mean
    err = sum(torch.abs(output-label))

    #mse
    mse_err = sum(pow(torch.abs(output - label),2))
    score_mse += mse_err
    #ptec
    score_ptecc += torch.abs(output - label)
    denominator += label

    #mne
    mne_estimated += output
    mne_label += label
    if i==0 :
        score_mae = err
    else :
        score_mae = torch.cat((score_mae, err), 0)

score_ptecc /= 2 * denominator
score_ptecc = (1 - score_ptecc) * 100
score_mae = score_mae.mean()
score_mne = torch.abs(mne_estimated - mne_label)/denominator
score_mse /= count
print('Proportion of energy correctly classified: ', score_ptecc[0][0], '%')
print('Mean absolute error: ', score_mae)
print('Mean normalised error: ', score_mne[0][0])
print('Mean Squared Error: ', score_mse[0])
print('Root Mean Squared Error: ', torch.sqrt(score_mse)[0])
#Ploting some random test examples - visualization of neural network's results
dataiter = iter(microwave_testloader)
for i in range(30):
    aggregate, labels = dataiter.next()
    label = labels.float().mean()
    inputs = aggregate
    inputs = inputs.view(1, -1, MICROWAVE_WINDOW_SIZE)
    if torch.cuda.is_available():
        inputs, labels = inputs.cuda(), labels.cuda
    inputs = Variable(inputs.float())
    outputs = net(inputs)
    show_output(aggregate=aggregate[0], individual=labels[0], output=outputs.data[0][0], label=label, window_size=MICROWAVE_WINDOW_SIZE)
