# Based on https://www.kaggle.com/parasjindal96/how-to-normalize-dataframe-pandass
import pandas as pd
from sklearn import preprocessing
from data_treatment import DataAtts
import matplotlib.pyplot as plt


file_names=["original_data/creditcard.csv", "original_data/data.csv", "original_data/diabetes.csv"]
for name in file_names:
    dataAtts = DataAtts(name)    
    data = pd.read_csv(name)


    dataNorm=((data-data.min())/(data.max()-data.min()))
    dataNorm[dataAtts.class_name]=data[dataAtts.class_name]
    dataNorm.to_csv( "original_data/" + dataAtts.fname + "_escalonated.csv", index=False)

# data['Pregnancies'].plot(kind='bar')
# plt.show()