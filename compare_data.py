import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm
import ipywidgets as widgets
import matplotlib.pyplot as plt
import glob
from data_treatment import DataAtts
from IPython.display import display
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import export_graphviz # Decision tree from sklearn
import pydotplus # Decision tree plotting

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
        
def create_comparing_table(original_data_name, fake_data_name):
    
    dataAtts = DataAtts(original_data_name)    
    data = pd.read_csv(original_data_name)
    fake_data = pd.read_csv(fake_data_name)
    fake_data.loc[getattr(fake_data, dataAtts.class_name) >= 0.5, dataAtts.class_name] = 1
    fake_data.loc[getattr(fake_data, dataAtts.class_name) < 0.5, dataAtts.class_name] = 0

    # Creates the training set
    training_data = [["original", data.head(int(data.shape[0]*0.7))]]
    fake_name = "fake" + str(fake_data_name).split("/")[2][0]
    training_data.append([fake_name, fake_data.head(int(fake_data.shape[0]*0.7))])
    
    test = data.tail(int(data.shape[0]*0.3))
    
    print("| Database \t| Proportion \t| Test Error \t|")
    print("| ---------\t| ---------: \t| :--------- \t|")

    for episode in training_data:
            name = episode[0]
            train = episode[1]
            try:
                positive=str(round(train[dataAtts.class_name].value_counts()[0]/len(train) * 100,2))
            except:
                positive="0"
            try:
                negative=str(round(train[dataAtts.class_name].value_counts()[1]/len(train) * 100,2))
            except:
                negative="0"


            trainX = train.drop(dataAtts.class_name, 1)
            testX = test.drop(dataAtts.class_name, 1)
            y_train = train[dataAtts.class_name]
            y_test = test[dataAtts.class_name]
            #trainX = pd.get_dummies(trainX)

            clf1 = DT(max_depth = 3, min_samples_leaf = 1)
            clf1 = clf1.fit(trainX,y_train)
            export_graphviz(clf1, out_file="models/tree.dot", feature_names=trainX.columns, class_names=["0","1"], filled=True, rounded=True)
            g = pydotplus.graph_from_dot_file(path="models/tree.dot")

            pred = clf1.predict_proba(testX)
            if pred.shape[1] > 1:
                pred = np.argmax(pred, axis=1)
            else:
                pred = pred.reshape((pred.shape[0]))
                if negative=="0":
                    pred = pred-1

            mse = round(((pred - y_test.values)**2).mean(axis=0), 4)

            string="| " + name + " \t| " + positive + "/" + negative + " \t| " + str(mse) + " \t|"
            print(string)

    
    