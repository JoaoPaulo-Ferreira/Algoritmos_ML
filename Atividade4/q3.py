import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from dtree import DTREE 
from sklearn.model_selection import KFold

data = np.loadtxt("trab4.data", dtype=np.float, delimiter=",")

kf = KFold(n_splits=5)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

total_acc = 0
clf = DTREE()
for train_idx, test_idx in kf.split(data):
    clf.fit(data[train_idx, 0:4], data[train_idx,4])
    y_pred = clf.predict(data[test_idx, 0:4])
    total_acc += accuracy(data[test_idx,4], y_pred)
total_acc /= 5.0

print ("Accuracy:", total_acc)