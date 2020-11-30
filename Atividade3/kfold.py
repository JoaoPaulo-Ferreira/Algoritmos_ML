import numpy as np
import matplotlib.pyplot as plt

def acc(y_real, y_pred):
    n = y_real.shape[0]
    n_true = np.count_nonzero(y_real == y_pred)
    return n_true/n 

def kfold(X, Y, metodo, k = 5):
    err=0
    step = (1/k) * X.shape[0]
    # print("step = ", step)
    masks = np.zeros((X.shape[0]))
    for i in range(k):
        masks[int(step*i):int(step*(i+1))] = True
        X_train = X[(masks==0)]
        y_train = Y[(masks==0)]
        X_test = X[(masks == 1)]
        y_test = Y[(masks==1)]
        masks = np.zeros((X.shape[0]))
        metodo.fit(X_train,y_train)
        y_pred = metodo.predict(X_test)
        err += acc(y_test, y_pred)
    return err/float(k)

