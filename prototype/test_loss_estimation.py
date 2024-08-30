import pytest
from loss_estimation import LossEstimator
import os
import random
from typing import Dict, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, Dataset, Subset
import nannyml as nml
from IPython.display import display
from PIL import Image



class MockNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10*10*3, 84)
        self.fc2 = nn.Linear(84, 3)
        self.softmax = torch.nn.Softmax()
    
    def forward(self,x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class SimpleDataset(datasets.VisionDataset):
    def __init__(self, images, labels, transform=None):
        self.transform = transform
        self.images = images
        self.labels = labels
        self.class_names = sorted(np.unique(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx]).float()

        label = torch.tensor(self.labels[idx]).float()

        return img, label

class MockDataset(SimpleDataset):
    def __init__(self):
        images = [
            np.zeros((3,10,10)),
            np.ones((3,10,10)),
            2*np.ones((3,10,10)),
            255*np.ones((3,10,10)),
        ]
        labels = [
            1,
            2,
            1,
            0
        ]
        super().__init__(images, labels)

class TestLEUnit():
    def test_eval_builds_dict(self):
        torch._dynamo.disable()
        torch._dynamo.config.suppress_errors = True

        ds = MockDataset()
        class_names = np.unique(ds.labels)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.compile(MockNet().to(device))
        model = cast(MockNet, model)
        le = LossEstimator()

        ds_dict = le._eval_model(model, ds, ds.class_names, True)
        assert sorted(ds_dict["y"]) == sorted(ds.labels)
