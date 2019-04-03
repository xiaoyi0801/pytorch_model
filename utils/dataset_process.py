#!/usr/bin/env python3.6

"""
data process api.
"""

import os
import random
import pandas as pd
import numpy as np


# generate csv file for training and validation.
def dataset_generate(data_dir):
    files = []
    label_set = set()
    for file in os.listdir(data_dir):
        if not file.endswith('.jpg'):
            continue
        label = file[:file.rfind('_')]
        label_set.add(label)
        # print(label)
        files.append([file, label])
    label_map = dict()
    for i, name in enumerate(label_set):
        label_map[name] = i
    random.shuffle(files)
    trainset = files[0: int(len(files) * 0.8)]
    valset = files[int(len(files) * 0.8):]
    for phase_name, phase in [["train_data", trainset], ["val_data", valset]]:
        image_name = list(map(lambda x: x[0], phase))
        label_name = list(map(lambda x: x[1], phase))
        label_id = list(map(lambda x: label_map[x[1]], phase))
        phase_set = dict(image=image_name, label_name=label_name, label=label_id)
        phase_df = pd.DataFrame(phase_set)
        phase_df.to_csv("../data/{}.csv".format(phase_name))


if __name__ == "__main__":
    data_dir = "/fastai/data/oxford-iiit-pet/images"
    dataset_generate(data_dir)

