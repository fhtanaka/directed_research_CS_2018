# Based on https://www.kaggle.com/parasjindal96/how-to-normalize-dataframe-pandass
import pandas as pd
from sklearn import preprocessing
from data_treatment import DataAtts
import matplotlib.pyplot as plt


file_name="original_data/data.csv"
dataAtts = DataAtts(file_name)    
data = pd.read_csv(file_name)


dataNorm=((data-data.min())/(data.max()-data.min()))
dataNorm[dataAtts.class_name]=data[dataAtts.class_name]
dataNorm.to_csv( "original_data/" + dataAtts.fname + "_escalonated.csv")

# data['Pregnancies'].plot(kind='bar')
# plt.show()