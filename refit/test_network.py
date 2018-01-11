import sys
sys.path.insert(0, '..')
import numpy as np
import torch
import refit_parameters as params
import refit_data as data
from dataset import *
from matplotlib import pyplot as plt
from transformations import Normalize
import torch.nn as nn
import torch.optim as optim
from models import *
from torch.autograd import Variable
import torchvision
from scores import *

test_individual_networks = {"House16":"Appliance2"}
#Reading mean and standard deviation features of all houses's aggregate consumption
with open('fridge_data_features.pickle', 'rb') as f:
    features = pickle.load(f)
mean, std = features['mean'], features['std']

#Reading data lengths of each house written inside of pickle binary file
with open('fridge_house_lengths.pickle','rb') as f:
    house_lengths = pickle.load(f)

for test, channel in test_individual_networks.items():
    print('Testing network for: ', test)
    testset_df = pd.read_csv('../data/CLEAN_REFIT/CLEAN_' + test + '.csv', skiprows=range(1,round(0.8 * house_lengths[test])))
    agg, iam = np.copy(np.array(testset_df['Aggregate'])), np.copy(np.array(testset_df[channel]))
    del testset_df
    agg, iam = data.create_windows(agg, iam, params.REFRIGERATOR_WINDOW_SIZE)
    testset = [agg, iam]
    testset = RefitDataset(testset)
    testset.init_transformation(torchvision.transforms.Compose([Normalize(mean=mean, sd=std)]))

    best_model = torch.load('models/refit_refrigerator_trained_model'+test+'2.pt')
    if torch.cuda.is_available():
        best_model = best_model.cuda()
    best_model.eval()
    best_scores= get_scores(best_model, testset, 1, params.REFRIGERATOR_WINDOW_SIZE, std, mean)
