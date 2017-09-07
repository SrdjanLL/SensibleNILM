import numpy as np
import dataset as redd
import torch
from parameters import *
import random
import matplotlib.pyplot as plt

#This is the demo ploting training examples from torch custom dataset for REDD refrigerator data.

dataset = redd.RefrigeratorDataSet("../data/REDD/")
for i in range(10):
    index = random.randint(0, dataset.__len__())
    print(index)
    agg, iam = dataset .__getitem__(index)
    y_agg = agg.numpy()
    x_agg = range(1, 4 * REFRIGERATOR_WINDOW_SIZE , 4)
    y_iam = iam.numpy()


    plt.plot(x_agg, y_agg, 'C1', label='Aggregate usage')
    plt.plot(x_agg, y_iam, 'C2', label='Individual usage')
    plt.show()
