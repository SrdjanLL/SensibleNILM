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
#Initialization of dataset
train_set, test_set = data.generate_clean_data(data_dir="data/REDD/", appliance='Microwave', window_size=params.MICROWAVE_WINDOW_SIZE, proportion=[15, 0], threshold=200, test=True, test_on='All') #stride=1

# initialization of custom pytorch datasets
microwave_train_set = dataset.REDDDataset(data=train_set)
microwave_test_set = dataset.REDDDataset(data=test_set)
#Getting mean and standard deviation so Normalization can be performed
#mean, std = microwave_train_set.get_mean_and_std()
mean = 444.516250434
std = 828.08954202
microwave_train_set.init_transformation(torchvision.transforms.Compose([Normalize(mean=mean, sd=std)]))
microwave_test_set.init_transformation(torchvision.transforms.Compose([Normalize(mean=mean, sd=std)]))
print('Training set size: ', len(microwave_train_set))
print('Test set size:', len(microwave_test_set))

microwave_trainloader = torch.utils.data.DataLoader(microwave_train_set, batch_size=32, num_workers=2)
microwave_testloader = torch.utils.data.DataLoader(microwave_test_set, batch_size=32, num_workers=2)

#Initialization of neural network
best_model = ConvMicroNILM()
try:
    best_model = torch.load('models/microwave_trained_model1.pt')
except FileNotFoundError:
    print('There is no pretrained model')
net = ConvMicroNILM()
if torch.cuda.is_available():
    net = net.cuda()
    net.load_state_dict(best_model.state_dict())
    best_model = best_model.cuda()
best_model.eval()
best_model_scores = scores.get_scores(best_model, microwave_test_set, 1, params.MICROWAVE_WINDOW_SIZE, std, mean)

criterion = nn.MSELoss()
optimimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)
print("Start of training: ")
for epoch in range(15):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(microwave_trainloader, 0):
        #Get the inputs
        if i == len(microwave_trainloader) - 1:
            continue
        inputs, labels = data
        label = labels.mean(dim=1).float()
        inputs = inputs.view(-1, 1, params.MICROWAVE_WINDOW_SIZE)
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
    new_scores = scores.get_scores(net, microwave_test_set, 1, params.MICROWAVE_WINDOW_SIZE, std, mean)
    if scores.compare_scores(best_scores, new_scores) > 0:
        best_model.load_state_dict(net.state_dict())
        best_scores = new_scores
        torch.save(best_model, 'models/microwave_trained_model1.pt')
        print('Best trained model')
    print('-------------------------------------------\n\n')
print('Finished Training')
torch.save(best_model, 'models/microwave_trained_model1.pt')
net.eval()
last_scores = scores.get_scores(net, microwave_test_set, 1, params.MICROWAVE_WINDOW_SIZE, std, mean)
dataiter = iter(microwave_testloader)
for i in range(len(microwave_testloader)):
    aggregate, labels = dataiter.next()
    label = labels.mean(dim=1).float()
    inputs = aggregate
    inputs = inputs.view(-1, 1, params.MICROWAVE_WINDOW_SIZE)
    if torch.cuda.is_available():
        inputs, labels = inputs.cuda(), labels.cuda()
    inputs = Variable(inputs.float())
    outputs = net(inputs)
    #print('Outputs:', outputs)
    for i in range(len(inputs)):
        data.show_output(aggregate=aggregate[i], individual=labels[i], output=outputs.data[i][0], label=label[i], window_size=params.MICROWAVE_WINDOW_SIZE)
