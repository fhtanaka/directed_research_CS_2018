import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import export_graphviz # Decision tree from sklearn

import pydotplus # Decision tree plotting
from IPython.display import Image

def create_and_evaluate_DT (training_data, testing_data, depth=5, min_leaves=1):
    trainX = training_data.iloc[:,:-1]
    testX = testing_data.iloc[:,:-1]
    y_train = training_data["Outcome"]
    y_test = testing_data["Outcome"]

    clf1 = DT(max_depth = depth, min_samples_leaf=min_leaves)
    clf1 = clf1.fit(trainX, y_train)

    export_graphviz(clf1, out_file="tree.dot", feature_names=trainX.columns, class_names=["0","1"], filled=True, rounded=True)
    g = pydotplus.graph_from_dot_file(path="tree.dot")
    Image(g.create_png())

    pred = clf1.predict_proba(testX)
    pred = np.argmax(pred, axis=1)
    mse = ((pred - y_test.values)**2).mean(axis=0)
    print("Prediction error: ", mse)