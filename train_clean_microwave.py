import numpy as np
import torch
from parameters import *
import data
from dataset import *
from matplotlib import pyplot as plt
from transformations import Normalize
import torch.nn as nn
import torch.optim as optim
from models import *
from torch.autograd import Variable
import torchvision

def show_example(aggregate, individual):
    plt.plot(range(1, 4 * MICROWAVE_WINDOW_SIZE, 4), aggregate.numpy(), 'C1', label='Aggregate usage')
    plt.plot(range(1, 4 * MICROWAVE_WINDOW_SIZE, 4), individual.numpy(), 'C2', label='Individual usage')
    plt.tight_layout()
    plt.legend()
    plt.show()

def show_output(aggregate, individual, output, label):
    if torch.cuda.is_available():
        aggregate= aggregate.cpu()
        individual = individual.cpu()
    plt.plot(range(1, 4 * MICROWAVE_WINDOW_SIZE, 4), aggregate.numpy(), 'C1', label='Aggregate usage')
    plt.plot(range(1, 4 * MICROWAVE_WINDOW_SIZE, 4), individual.numpy(), 'C2', label='Individual usage')
    plt.plot(range(1, 4 * MICROWAVE_WINDOW_SIZE, 4), [label for i in range(MICROWAVE_WINDOW_SIZE)], 'C3', label='Real average usage')
    plt.plot(range(1, 4 * MICROWAVE_WINDOW_SIZE, 4), [output for i in range(MICROWAVE_WINDOW_SIZE)], 'C4', label='Predicted average usage')

    plt.tight_layout()
    plt.legend()
    plt.show()

#Initialization of dataset
microwave_train_set = REDDCleanDataset(data_dir="data/REDD/", appliance='Microwave', window_size=MICROWAVE_WINDOW_SIZE)

#Getting mean and standard deviation so Normalization can be performed
mean, std = microwave_train_set.get_mean_and_std()
microwave_train_set.init_transformation(torchvision.transforms.Compose([Normalize(mean=mean, sd=std)]))
print('Dataset size: ', len(microwave_train_set))

#Splitting dataset into training and test set
train_examples_count = round(len(microwave_train_set) * 0.8)
microwave_trainloader = torch.utils.data.DataLoader(microwave_train_set, batch_size=1, sampler=torch.utils.data.sampler.SubsetRandomSampler(range(train_examples_count)), num_workers=2)
microwave_testloader = torch.utils.data.DataLoader(microwave_train_set, batch_size=1, sampler=torch.utils.data.sampler.SubsetRandomSampler(range(train_examples_count, len(microwave_train_set))),num_workers=2)

#Initialization of neural network
net = ConvMicroNILM()
if torch.cuda.is_available():
    net = net.cuda()
criterion = nn.MSELoss()
optimimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)
print("Start of training: ")
for epoch in range(2):
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
            
print('Finished Training')

score_mae = None

for i, data in enumerate(microwave_testloader, 0):
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

    if i==0 :
        score_mae = err
    else :
        score_mae = torch.cat((score_mae, err), 0)

score_mae = score_mae.mean()
print('Mean absolute error: ', score_mae)

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
    show_output(aggregate=aggregate[0], individual=labels[0], output=outputs.data[0][0], label=label)
