import torch
import pandas as pd
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from data_treatment import DataSet, DataAtts
from discriminator import *
from generator import *
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import glob


db_names=["diabetes_escalonated", "data_escalonated", "creditcard_escalonated"]

for name in db_names:
    original_db_name = name
    original_db_path = "original_data/" + original_db_name + ".csv"
    original_db = pd.read_csv(original_db_path)
    original_db_size=original_db.shape[0]

    folder_name="models/" + original_db_name + "/generator*"
    for file in glob.glob(folder_name):
        name = file.split("/")[-1][10:-4]
        print(name)
        try:
            checkpoint= torch.load(file, map_location='cuda')
        except:
            checkpoint= torch.load(file, map_location='cpu')
        generator = GeneratorNet(**checkpoint['model_attributes'])
        generator.load_state_dict(checkpoint['model_state_dict'])
        size = original_db_size
        new_data = generator.create_data(100)
        df = pd.DataFrame(new_data, columns=original_db.columns)
        name = name + "_size-" + str(size)
        df.to_csv( "fake_data/" + original_db_name + "/" + name + ".csv", index=False)