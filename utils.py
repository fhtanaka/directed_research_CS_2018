import torch
import pandas as pd
from torch.autograd.variable import Variable
from data_treatment import DataSet, DataAtts
import os

def random_noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): 
        return n.cuda() 
    return n

def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def escalonate_creditcard_db():
    if (os.path.isfile('./original_data/creditcard.csv') and not os.path.isfile('./original_data/creditcard_escalonated.csv')):
        name="original_data/creditcard.csv"
        dataAtts = DataAtts(name)    
        data = pd.read_csv(name)
        dataNorm=((data-data.min())/(data.max()-data.min()))
        dataNorm[dataAtts.class_name]=data[dataAtts.class_name]
        dataNorm.to_csv( "original_data/" + dataAtts.fname + "_escalonated.csv", index=False)

