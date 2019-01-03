import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm
import ipywidgets as widgets
import matplotlib.pyplot as plt
import glob
from data_treatment import DataAtts
from IPython.display import display

def compare_data (original_data, fake_data, size_of_fake, mode="save"):
    dataAtts = DataAtts(original_data)
    
    data = pd.read_csv(original_data)
    fake_data = pd.read_csv(fake_data).tail(size_of_fake)
    print(dataAtts.message, "\n")
    print(dataAtts.values_names[0], round(data[dataAtts.class_name].value_counts()[0]/len(data) * 100,2), '%  of the dataset')
    print(dataAtts.values_names[1], round(data[dataAtts.class_name].value_counts()[1]/len(data) * 100,2), '%  of the dataset')
    
    classes = list(data)

    for name in classes:
        if name=="Unnamed: 32":
            continue

        plt.xlabel('Values')
        plt.ylabel('Probability')
        plt.title(name + " distribution")
        real_dist = data[name].values
        fake_dist = fake_data[name].values
        plt.hist(real_dist, 50, density=True, alpha=0.5)
        plt.hist(fake_dist, 50, density=True, alpha=0.5, facecolor='r')
        if mode=="save":
            plt.savefig('fake_data/'+ dataAtts.fname + "/"+name+'_distribution.png')
        elif mode=="show":
            plt.show()
        plt.clf()
    
    