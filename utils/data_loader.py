#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils, models


class Pet_Dataset(Dataset):
    def __init__(self, image_path, csv_path, phase="train", transform=None):
        super(Pet_Dataset, self).__init__()
        assert phase in ("train", "val"), "error phase name: {}, must be 'train' or 'val'".format(phase)
        self.image_path = image_path
        self.csv_path = csv_path
        self.phase = phase
        self.dataset = pd.read_csv(os.path.join(csv_path, "{}_data.csv".format(phase)))
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_name = os.path.join(self.image_path, self.dataset.iloc[idx, 1])
        image = Image.open(image_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.dataset.iloc[idx, 3]
        return image, label

