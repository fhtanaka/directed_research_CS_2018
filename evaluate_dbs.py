import numpy as np
import pandas as pd
from data_treatment import DataAtts

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import export_graphviz # Decision tree from sklearn

# import pydotplus # Decision tree plotting
# from IPython.display import Image

# import ipywidgets as widgets
import glob


file_names=["original_data/diabetes_escalonated.csv", "original_data/data_escalonated.csv", "original_data/creditcard_escalonated.csv"]
report_names=["values_analysis/diabetes_escalonated_db_analysis.txt", "values_analysis/data_escalonated_db_analysis.txt", "values_analysis/creditcard_escalonated_db_analysis.txt"]
for file_name, report_name in zip(file_names, report_names):
    dataAtts = DataAtts(file_name)    
    data = pd.read_csv(file_name)
    folder_name = file_name[14:-4]

    # Creates the training set
    training_data = [["original", data.head(int(data.shape[0]*0.7))]]
    test = data.tail(int(data.shape[0]*0.3))
    for file in glob.glob("fake_data/" + folder_name + "/*.csv"):
        name = "fake" + str(file).split("/")[2][2:]
        fake_data = pd.read_csv(file)
        fake_data.loc[getattr(fake_data, dataAtts.class_name) >= 0.5, dataAtts.class_name] = 1
        fake_data.loc[getattr(fake_data, dataAtts.class_name) < 0.5, dataAtts.class_name] = 0
        fake_training=fake_data.head(int(fake_data.shape[0]*0.7))
        training_data.append([name, fake_training])

    report = open(report_name, "w")
    print("\n\n" + file_name)
    print("| Database \t| Proportion \t| Test Error \t|")
    print("| ---------\t| ---------: \t| :--------- \t|")

    report.write(file_name+"\n")
    report.write("| Database \t| Proportion \t| Test Error \t|\n")
    report.write("| ---------\t| ---------: \t| :--------- \t|\n")

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
            # g = pydotplus.graph_from_dot_file(path="models/tree.dot")

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
            report.write(string+"\n")