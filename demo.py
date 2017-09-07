import numpy as np
import dataset as redd
import torch
from parameters import *
import random
import matplotlib.pyplot as plt

#This is the demo ploting training examples from torch custom dataset for REDD refrigerator data.

refrigerator_dataset = redd.RefrigeratorREDDDataSet("data/REDD/")
microwave_dataset = redd.MicrowaveREDDDataSet("data/REDD/")
dishwasher_dataset = redd.DishwasherREDDDataSet("data/REDD/")

for i in range(30):
    dataset_rand = random.randint(1, 100)
    dataset = None
    title = ""
    window_size = 0
    if dataset_rand % 3 == 0:
        dataset = refrigerator_dataset
        title = "Refrigerator"
        window_size = REFRIGERATOR_WINDOW_SIZE
    elif dataset_rand % 3 == 1:
        dataset = microwave_dataset
        title = "Microwave"
        window_size = MICROWAVE_WINDOW_SIZE
    else:
        dataset = dishwasher_dataset
        title = "Dishwasher"
        window_size = DISHWASHER_WINDOW_SIZE
    index = random.randint(0, dataset.__len__())
    print(index)
    agg, iam = dataset .__getitem__(index)
    y_agg = agg.numpy()
    x_agg = range(1, 4 * window_size , 4)
    y_iam = iam.numpy()

    plt.title(title)
    plt.tight_layout()
    plt.plot(x_agg, y_agg, 'C1', label='Aggregate usage')
    plt.plot(x_agg, y_iam, 'C2', label='Individual usage')
    plt.legend()
    plt.show()
