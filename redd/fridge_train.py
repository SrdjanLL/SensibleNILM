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
#Creating custom dataset
train_set, test_set = data.generate_clean_data(data_dir="data/REDD/", appliance='Refrigerator', window_size=params.REFRIGERATOR_WINDOW_SIZE, proportion=[0, 0], threshold=80, test=True, test_on='All') #stride=5

# initialization of custom pytorch datasets
refrigerator_train_set = dataset.REDDDataset(data=train_set)
refrigerator_test_set = dataset.REDDDataset(data=test_set)
#Getting mean and standard deviation so Normalization can be performed
mean, std = refrigerator_train_set.get_mean_and_std()
refrigerator_train_set.init_transformation(torchvision.transforms.Compose([Normalize(mean=mean, sd=std)]))
refrigerator_test_set.init_transformation(torchvision.transforms.Compose([Normalize(mean=mean, sd=std)]))
print('Training set size: ', len(refrigerator_train_set))
print('Test set size: ', len(refrigerator_test_set))
#Splitting dataset into training and test set
refrigerator_trainloader = torch.utils.data.DataLoader(refrigerator_train_set, batch_size=32, num_workers=2)
refrigerator_testloader = torch.utils.data.DataLoader(refrigerator_test_set, batch_size=32,num_workers=2)

#Initialization of neural network
best_model = ConvFridgeNILM()
try:
    best_model = torch.load('models/refrigerator_trained_model.pt')
except FileNotFoundError:
    print('There is no pretrained model.')

net = ConvFridgeNILM()
if torch.cuda.is_available():
    net = best_model.cuda()
    best_model = best_model.cuda()
best_model.eval()
best_model_scores = scores.get_scores(best_model, refrigerator_test_set, 1, params.REFRIGERATOR_WINDOW_SIZE, std, mean)


#Defining loss function and optimizer
criterion = nn.MSELoss()
optimimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)

print("Start of training: ")
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(refrigerator_trainloader, 0):
        #Get the inputs
        if i == len(refrigerator_trainloader) - 1:
            continue
        inputs, labels = data
        label = labels.mean(dim=1).float()
        inputs = inputs.view(-1, 1, params.REFRIGERATOR_WINDOW_SIZE)
        if torch.cuda.is_available():
            inputs, labels, label = inputs.cuda(), labels.cuda(), label.cuda()

        #Wrapping into torch.autograd.Variables - required for pytorch framework so that backpropagation can be performed
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
        if i % 500 == 499:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0

    net.eval()
    new_scores = scores.get_scores(net, refrigerator_test_set, 1, params.REFRIGERATOR_WINDOW_SIZE, std, mean)
    if scores.compare_scores(best_scores, new_scores) > 0:
        best_model.load_state_dict(net.state_dict())
        best_scores = new_scores
        torch.save(best_model, 'models/refrigerator_trained_model.pt')
        print('Best trained model')
    print('-------------------------------------------\n\n')
print('Finished Training')
torch.save(best_model, 'models/refrigerator_trained_model.pt')
net.eval()
last_scores = scores.get_scores(net, refrigerator_test_set, 1, params.REFRIGERATOR_WINDOW_SIZE, std, mean)
dataiter = iter(refrigerator_testloader)
for i in range(len(refrigerator_testloader)):
    aggregate, labels = dataiter.next()
    label = labels.mean(dim=1).float()
    inputs = aggregate
    inputs = inputs.view(-1, 1, params.REFRIGERATOR_WINDOW_SIZE)
    if torch.cuda.is_available():
        inputs, labels = inputs.cuda(), labels.cuda()
    inputs = Variable(inputs.float())
    outputs = net(inputs)
    #print('Outputs:', outputs)
    for i in range(len(inputs)):
        data.show_output(aggregate=aggregate[i], individual=labels[i], output=outputs.data[i][0], label=label[i], window_size=params.REFRIGERATOR_WINDOW_SIZE)
