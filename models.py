import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#First model that was trained, terrible
class LinearNILM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(LinearNILM, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ConvNILM(nn.Module):

    def __init__(self):
        super(ConvNILM, self).__init__()
        #First set of conv layers
        self.conv1 = nn.Conv1d(1, 16, 5)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 16, 3)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 16, 3)
        self.bn3 = nn.BatchNorm1d(16)
        self.pool = nn.AvgPool1d(4, stride=2)
        self.drop = nn.Dropout(p = 0.2)
        self.out1 = nn.Linear(16 * 296, 4096)

        #self.out1 = nn.Linear(16 * 36, 4096) #Refrigerator with 3 conv layers only
        self.out2 = nn.Linear(4096, 3072)
        self.out3 = nn.Linear(3072, 2048)
        self.out4 = nn.Linear(2048, 512)
        self.out5 = nn.Linear(512, 1)

    def num_flat_features(self, x):
        size = x.size()[1:] #all dimensions except the batch dimensions
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        #First to implement feedforward network with first set of conv layers
        x = self.pool(self.bn1(self.conv1(x)))
        x = self.pool(self.bn2(self.conv2(x)))
        x = self.pool(self.bn3(self.conv3(x)))
        #print('Size of x after 3 convolutional layers: ', x.size())

        x = x.view(-1, self.num_flat_features(x))
        #x = self.drop(F.relu(self.out1(x)))
        x = F.relu(self.out1(x))
        x = F.relu(self.out2(x))
        x = F.relu(self.out3(x))
        x = F.relu(self.out4(x))
        x = self.out5(x)
        #x = self.out3(x)

        return x

class ConvDishNILM(nn.Module):

    def __init__(self):
        super(ConvDishNILM, self).__init__()
        #First set of conv layers
        self.conv1 = nn.Conv1d(1, 16, 5)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 16, 3)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 16, 3)
        self.bn3 = nn.BatchNorm1d(16)
        self.pool = nn.AvgPool1d(4, stride=2)
        self.drop = nn.Dropout(p = 0.2)
        self.out1 = nn.Linear(16 * 75, 4096)

        #self.out1 = nn.Linear(16 * 36, 4096) #Refrigerator with 3 conv layers only
        self.out2 = nn.Linear(4096, 3072)
        self.out3 = nn.Linear(3072, 2048)
        self.out4 = nn.Linear(2048, 512)
        self.out5 = nn.Linear(512, 1)

    def num_flat_features(self, x):
        size = x.size()[1:] #all dimensions except the batch dimensions
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        #First to implement feedforward network with first set of conv layers
        x = self.pool(self.bn1(self.conv1(x)))
        x = self.pool(self.bn2(self.conv2(x)))
        x = self.pool(self.bn3(self.conv3(x)))
        #print('Size of x after 3 convolutional layers: ', x.size())

        x = x.view(-1, self.num_flat_features(x))
        #x = self.drop(F.relu(self.out1(x)))
        x = F.relu(self.out1(x))
        x = F.relu(self.out2(x))
        x = F.relu(self.out3(x))
        x = F.relu(self.out4(x))
        x = self.out5(x)
        #x = self.out3(x)

        return x

class ConvMicroNILM(nn.Module):

    def __init__(self):
        super(ConvMicroNILM, self).__init__()
        #First set of conv layers
        self.conv1 = nn.Conv1d(1, 16, 5)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 16, 3)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 16, 3)
        self.bn3 = nn.BatchNorm1d(16)
        self.pool = nn.AvgPool1d(4, stride=2)
        self.drop = nn.Dropout(p = 0.2)
        self.out1 = nn.Linear(16 * 21, 4096)

        #self.out1 = nn.Linear(16 * 36, 4096) #Refrigerator with 3 conv layers only
        self.out2 = nn.Linear(4096, 3072)
        self.out3 = nn.Linear(3072, 2048)
        self.out4 = nn.Linear(2048, 512)
        self.out5 = nn.Linear(512, 1)

    def num_flat_features(self, x):
        size = x.size()[1:] #all dimensions except the batch dimensions
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        #First to implement feedforward network with first set of conv layers
        x = self.pool(self.bn1(self.conv1(x)))
        x = self.pool(self.bn2(self.conv2(x)))
        x = self.pool(self.bn3(self.conv3(x)))
        #print('Size of x after 3 convolutional layers: ', x.size())

        x = x.view(-1, self.num_flat_features(x))
        #x = self.drop(F.relu(self.out1(x)))
        x = F.relu(self.out1(x))
        x = F.relu(self.out2(x))
        x = F.relu(self.out3(x))
        x = F.relu(self.out4(x))
        x = self.out5(x)
        #x = self.out3(x)

        return x
