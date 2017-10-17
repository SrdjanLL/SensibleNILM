import data
from matplotlib import pyplot as plt
import random
from parameters import *
aggregate_data, iam_data = data.generate_clean_data("data/REDD/", appliance='Microwave', window_size=MICROWAVE_WINDOW_SIZE)
print('Dataset length: ',len(aggregate_data))
for i in range(45):
    index = random.randint(0, len(aggregate_data))
    aggregate, iam = aggregate_data[index], iam_data[index]
    plt.plot(range(1, MICROWAVE_WINDOW_SIZE * 4, 4), aggregate, 'C1', label='Input aggregate')
    plt.plot(range(1, MICROWAVE_WINDOW_SIZE * 4, 4), iam, 'C2', label='Iam')
    plt.tight_layout()
    plt.legend()
    plt.show()
