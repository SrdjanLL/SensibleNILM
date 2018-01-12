import sys
sys.path.insert(0, '..')
import numpy as np
import torch
import refit_parameters as params
import refit_data as data
import dataset as datasets
from matplotlib import pyplot as plt
from transformations import Normalize
import torch.nn as nn
import torch.optim as optim
from models import *
from torch.autograd import Variable
import torchvision
import scores
import pickle
import pandas as pd
#Choosing on which houses network will be tested
test = ['House4']

#Reading mean and standard deviation features of all houses's aggregate consumption
with open('aggregate_features.pickle', 'rb') as f:
    features = pickle.load(f)
mean, std = features['mean'], features['std']

#Reading data lengths of each house written inside of pickle binary file
with open('house_lengths.pickle','rb') as f:
    house_lengths = pickle.load(f)

#house_train_length = {house: round(0.8 * length) for house, length in list(house_lengths.items())}

#For each house that is contained in test list only first 80% of the data is taken for training while last 20% remaining are for testing purposes
house_train_length = house_lengths.copy()
for house in test:
    house_train_length[house] = round(0.8 * house_train_length[house])

#Creating lists containing aggregate and individual power consumption of targeted appliance in numpy matrix format where each row is one training/testing example
#After agg and iam lists are created they are passed to an object of RefitDataset class
testset_df = pd.read_csv('../data/CLEAN_REFIT/CLEAN_' + test[0] + '.csv', skiprows=range(1,round(0.8 * house_lengths[test[0]])))
agg, iam = np.copy(np.array(testset_df['Aggregate'])), np.copy(np.array(testset_df[params.fridge_channels[test[0]]]))
del testset_df
agg, iam = data.create_windows(agg, iam, params.REFRIGERATOR_WINDOW_SIZE)
testset = [agg, iam]
testset = datasets.RefitDataset(testset)
# for i in range(10):
#     agg, iam = testset[i]
#     show_example(agg, iam, window_size=params.REFRIGERATOR_WINDOW_SIZE)
#Setting up Normalization transformation of each item returned by the testset object
testset.init_transformation(torchvision.transforms.Compose([Normalize(mean=mean, sd=std)]))

#Initialization of neural networks where best model would be pretrained model saved in models/refit_refrigerator_trained_model.pt file and net is newly created network
best_model = RefitConvFridgeNILM2()
# try:
#     best_model = torch.load('models/refit_refrigerator_trained_model3.pt')
# except FileNotFoundError:
#     print('There is no pretrained model.')
#     best_model = RefitConvFridgeNILM2()
net = RefitConvFridgeNILM2()

#If cuda is available everything will be processed on cuda
if torch.cuda.is_available():
    net = net.cuda()
    best_model = best_model.cuda()

#Evaluation of best model - before start of training
# best_model.eval()
# best_scores= get_scores(best_model, testset, 1, params.REFRIGERATOR_WINDOW_SIZE, std, mean)

#Mean Squared error is chosen as a loss function
criterion = nn.MSELoss()

#Stohastic Gradient Descent with learning rate 0.001 and momentum 0.9
optimimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)

# training_houses = {"House18":"Appliance1", "House19":"Appliance1","House20":"Appliance1","House21":"Appliance1"}
training_houses = {"House4":"Appliance1"}
# house, channel = 'House1', 'Appliance1'
#Training starts here
test_index = 0
print("Start of training: ")
for house in training_houses.keys():
    channel = training_houses[house]
    best_model = RefitConvFridgeNILM2()
    try:
        best_model = torch.load('models/refit_refrigerator_trained_model'+house+'.pt')
    except FileNotFoundError:
        print('There is no pretrained model.')
        best_model = RefitConvFridgeNILM2()
    net = RefitConvFridgeNILM2()

    #If cuda is available everything will be processed on cuda
    if torch.cuda.is_available():
        net = net.cuda()
        best_model = best_model.cuda()

    testset_df = pd.read_csv('../data/CLEAN_REFIT/CLEAN_' + test[test_index] + '.csv', skiprows=range(1,round(0.8 * house_lengths[test[test_index]])))
    agg, iam = np.copy(np.array(testset_df['Aggregate'])), np.copy(np.array(testset_df[params.fridge_channels[test[test_index]]]))
    del testset_df
    agg, iam = data.create_windows(agg, iam, params.REFRIGERATOR_WINDOW_SIZE)
    testset = [agg, iam]
    testset = datasets.RefitDataset(testset)
    test_index +=1
    # for i in range(10):
    #     agg, iam = testset[i]
    #     show_example(agg, iam, window_size=params.REFRIGERATOR_WINDOW_SIZE)
    #Setting up Normalization transformation of each item returned by the testset object
    testset.init_transformation(torchvision.transforms.Compose([Normalize(mean=mean, sd=std)]))
    #Evaluation of best model - before start of training
    # best_model.eval()
    # best_scores= scores.get_scores(best_model, testset, 1, params.REFRIGERATOR_WINDOW_SIZE, std, mean)

    #Mean Squared error is chosen as a loss function
    criterion = nn.MSELoss()

    #Stohastic Gradient Descent with learning rate 0.001 and momentum 0.9
    optimimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)
    for epoch in range(1):
        net.train()

        #House indicators indicate current positions within .csv files of eache house included in training set
        house_indicators = {house : 1 for house in training_houses.keys()}
        #Dictionary containing house names and channels where targeted appliance's data is contained
        channels = training_houses.copy()
        running_loss = 0.0
        batches_processed=0
        count = 160000
            #when training on multiple houses those house indicators should be randomly chosen and randomly generate subsets for training.
        while house_indicators[house] + params.REFRIGERATOR_WINDOW_SIZE <= house_train_length[house]:
            print('Reading')
            # window = pd.read_csv('data/CLEAN_REFIT/CLEAN_' + house + '.csv', skiprows=range(1, house_indicators[house]), nrows=params.REFRIGERATOR_WINDOW_SIZE, usecols=['Aggregate', channel])
            indicator, dataset = data.house_subset('../data/CLEAN_REFIT/CLEAN_' + house + '.csv', house_train_length[house], channel, params.REFRIGERATOR_WINDOW_SIZE, count=count, stride=10, readfrom=house_indicators[house])
            house_indicators[house] += indicator
            dataset = datasets.RefitDataset(dataset)
            dataset.init_transformation(torchvision.transforms.Compose([Normalize(mean=mean, sd=std)]))
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4)
            for i, data in enumerate(train_loader, 0):
                #Get the inputs
                inputs, labels = data
                label = labels.mean(dim=1).float()
                inputs = inputs.view(-1, 1, params.REFRIGERATOR_WINDOW_SIZE)
                if(inputs.size(dim=0) != 32):
                    continue
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
                batches_processed +=1
                if not batches_processed % 500:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, batches_processed, running_loss / 500))
                    running_loss = 0.0



        #Evaluation of trained network's performance and comparison against best trained model so far. If new network beats best model the new network is saved as the best model
        new_scores = scores.get_scores(net, testset, 1, params.REFRIGERATOR_WINDOW_SIZE, std, mean)
        if scpres.compare_scores(best_scores, new_scores) > 0:
            best_model.load_state_dict(net.state_dict())
            best_scores = new_scores.copy()
            torch.save(best_model, 'models/refit_refrigerator_trained_model'+house+'.pt')
            print('Best trained model')

print('Finished training.')
net.eval()
plot_test_set(net, testset, params.REFRIGERATOR_WINDOW_SIZE)
