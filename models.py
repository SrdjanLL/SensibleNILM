import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ConvFridgeNILM(nn.Module):
    def __init__(self):
        super(ConvFridgeNILM, self).__init__()
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
        self.drop = nn.Dropout(p = 0.2, inplace=True)
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
        x = F.relu(self.pool(self.bn1(self.conv1(x))))
        x = F.relu(self.pool(self.bn2(self.conv2(x))))
        x = F.relu(self.pool(self.bn3(self.conv3(x))))
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
        self.pool = nn.MaxPool1d(4, stride=2)
        self.drop = nn.Dropout(p = 0.2, inplace=True)
        self.out1 = nn.Linear(16 * 33, 4096)

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
        x = F.relu(self.pool(self.bn1(self.conv1(x))))
        x = F.relu(self.pool(self.bn2(self.conv2(x))))
        x = F.relu(self.pool(self.bn3(self.conv3(x))))
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

class RefitConvFridgeNILM(nn.Module):
    def __init__(self):
        super(RefitConvFridgeNILM, self).__init__()
        #First set of conv layers
        self.conv1 = nn.Conv1d(1, 32, 5)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, 3)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, 3)
        self.bn3 = nn.BatchNorm1d(32)
        self.pool = nn.AvgPool1d(4, stride=2)
        self.drop = nn.Dropout(p = 0.2)
        self.out1 = nn.Linear(32 * 146, 4096)

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

class RefitConvFridgeNILM2(nn.Module):
    def __init__(self):
        super(RefitConvFridgeNILM2, self).__init__()
        #First set of conv layers
        self.conv1 = nn.Conv1d(1, 32, 5)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 16, 3)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 16, 3)
        self.bn3 = nn.BatchNorm1d(16)
        self.pool = nn.AvgPool1d(4, stride=2)
        self.drop = nn.Dropout(p = 0.2)
        self.out1 = nn.Linear(16 * 146, 4096)

        #self.out1 = nn.Linear(16 * 36, 4096) #Refrigerator with 3 conv layers only
        self.out2 = nn.Linear(4096, 4096)
        self.out3 = nn.Linear(4096, 4096)
        self.out4 = nn.Linear(4096, 4096)
        self.out5 = nn.Linear(4096, 1)

    def num_flat_features(self, x):
        size = x.size()[1:] #all dimensions except the batch dimensions
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        #First to implement feedforward network with first set of conv layers
        x = F.relu(self.pool(self.bn1(self.conv1(x))))
        x = F.relu(self.pool(self.bn2(self.conv2(x))))
        x = F.relu(self.pool(self.bn3(self.conv3(x))))
        # print('Size of x after 3 convolutional layers: ', x.size())

        x = x.view(-1, self.num_flat_features(x))
        #x = self.drop(F.relu(self.out1(x)))
        x = F.relu(self.out1(x))
        x = F.relu(self.out2(x))
        x = F.relu(self.out3(x))
        x = F.relu(self.out4(x))
        x = self.out5(x)
        #x = self.out3(x)

        return x
