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
# dishwasher_train_set = dataset.REDDCleanDataset(data_dir="data/REDD/", appliance='Dishwasher', window_size=params.DISHWASHER_WINDOW_SIZE, proportion=[3, 900], threshold=10)
# dishwasher_test_set = dataset.REDDCleanDataset(data_dir="data/REDD/", appliance='Dishwasher', window_size=params.DISHWASHER_WINDOW_SIZE, test=True)
train_set, test_set = data.generate_clean_data(data_dir="../data/REDD/", appliance='Dishwasher', window_size=params.DISHWASHER_WINDOW_SIZE, proportion=[3, 0], threshold=10, test=True, test_on='All', stride=1)

# initialization of custom pytorch datasets
dishwasher_train_set = dataset.REDDDataset(data=train_set)
dishwasher_test_set = dataset.REDDDataset(data=test_set)
mean = 395.890660893
std = 615.928606564

dishwasher_train_set.init_transformation(torchvision.transforms.Compose([Normalize(mean=mean, sd=std)]))
dishwasher_test_set.init_transformation(torchvision.transforms.Compose([Normalize(mean=mean, sd=std)]))
print('Training set size: ', len(dishwasher_train_set))
print('Test set size:', len(dishwasher_test_set))
dishwasher_trainloader = torch.utils.data.DataLoader(dishwasher_train_set, batch_size=32, num_workers=8)
dishwasher_testloader = torch.utils.data.DataLoader(dishwasher_test_set, batch_size=32, num_workers=8)

best_model = ConvDishNILM()
try:
    best_model = torch.load('models/dishwasher_lucky_shot.pt')
except FileNotFoundError:
    print('There is no pretrained model')
best_model.eval()
net = ConvDishNILM()
if torch.cuda.is_available():
    net = net.cuda()
    best_model = best_model.cuda()

best_scores = scores.get_scores(best_model, dishwasher_test_set, 1, params.DISHWASHER_WINDOW_SIZE, std, mean)
criterion = nn.MSELoss()
optimimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)

print("Start of training: ")
for epoch in range(10):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(dishwasher_trainloader, 0):
        #Get the inputs
        if i == len(dishwasher_trainloader) - 1:
            continue
        inputs, labels = data
        label = labels.mean(dim=1).float()
        inputs = inputs.view(-1, 1, params.DISHWASHER_WINDOW_SIZE)
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
    new_scores = scores.get_scores(net, dishwasher_test_set, 32, params.DISHWASHER_WINDOW_SIZE, std, mean)
    if scores.compare_scores(best_scores, new_scores) > 0:
        best_model.load_state_dict(net.state_dict())
        best_scores = new_scores
        torch.save(best_model, 'models/dishwasher_base_model.pt')
        print('Best trained model')

    print('-------------------------------------------\n\n')
print('Finished Training')

print('Evaluation of current network: ')
net.eval()
dishwasher_testloader = torch.utils.data.DataLoader(dishwasher_test_set, batch_size=32, num_workers=1)


last_scores = scores.get_scores(net, dishwasher_test_set, 32, params.DISHWASHER_WINDOW_SIZE, std, mean)

# #Ploting test examples - visualization of neural network's results

dataiter = iter(dishwasher_testloader)
for i in range(len(dishwasher_testloader)):
    aggregate, labels = dataiter.next()
    label = labels.mean(dim=1).float()
    inputs = aggregate
    inputs = inputs.view(-1, 1, params.DISHWASHER_WINDOW_SIZE)
    if torch.cuda.is_available():
        inputs, labels = inputs.cuda(), labels.cuda()
    inputs = Variable(inputs.float())
    outputs = net(inputs)
    #print('Outputs:', outputs)
    for i in range(len(inputs)):
        data.show_output(aggregate=aggregate[i], individual=labels[i], output=outputs.data[i][0], label=label[i], window_size=params.DISHWASHER_WINDOW_SIZE)
