import torch
import pandas as pd
import os
from sklearn import preprocessing
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

class DataSet(Dataset):
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

class DataAtts():
    def __init__(self, file_name):
        if file_name == "original_data/data.csv":
            self.message = "Breast Cancer Wisconsin (Diagnostic) Data Set"
            self.class_name = "diagnosis"
            self.values_names = {0: "Benign", 1: "Malignant"}
            self.class_len = 32
            self.fname="data"
        elif file_name == "original_data/creditcard.csv":
            self.message = "Credit Card Fraud Detection"
            self.class_name = "Class"
            self.values_names = {0: "No Frauds", 1: "Frauds"}
            self.class_len = 31
            self.fname="creditcard"
        elif file_name == "original_data/diabetes.csv":
            self.message="Pima Indians Diabetes Database"
            self.class_name = "Outcome"
            self.values_names = {0: "Normal", 1: "Diabets"}
            self.class_len = 9
            self.fname="diabetes"
        else:
            print("File not found, exiting")
            exit()


