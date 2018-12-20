import torch
import pandas as pd
import os
from sklearn import preprocessing
from utils import Logger
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets, utils
from torch.utils.data import Dataset, DataLoader

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # print (sample.values[2])
        # print (torch.from_numpy(sample.values)[2].item())
        return torch.from_numpy(sample.values)

class CreditCardDataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=transforms.Compose([ToTensor()])):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.data = pd.read_csv(csv_file).head(100000)
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        if self.transform:
            item = self.transform(item)
        return item

    def get_columns(self):
        return self.data.columns


