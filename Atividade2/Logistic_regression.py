import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import conf_matrix as cm

def acc(y_real, y_pred):
    n = y_real.shape[0]
    n_true = np.count_nonzero(y_real == y_pred)
    return n_true/n    

def normalizacao01(x):
    minx = np.amin(x)
    maxx = np.amax(x)
    return (x-minx)/(maxx-minx)

class rl:
    def __init__(self, n_inter, alpha):
        self.n_inter = n_inter
        self.alpha = alpha

    def logistic(self, X):
        return 1/(1 + np.exp(-np.dot(X, self.B.T)))

    def J(self, X, y):
        error = 0.
        for xi,yi in zip(X,y):
            yi_pred = self.logistic(xi)
            error += (-yi*np.log(yi_pred) - (1-yi)*np.log(1-yi_pred))
        return error

    def fit(self, X, y):
        
        n_samples, _ = X.shape
        X = np.c_[np.ones(n_samples),X]
        _, n_features = X.shape
        self.B = np.zeros(n_features)
        self.errors=[]
        # print(X)
        for _ in range(self.n_inter):
            y_pred = self.logistic(X)
            # print(y_pred)
            err = y - y_pred

            self.B += self.alpha * 1/n_samples* np.dot(err.T, X)
            self.errors.append(self.J(X, y))

    def predict(self, X):
        n_samples, _ = X.shape
        X = np.c_[np.ones(n_samples),X]
        y_pred = self.logistic(X)
        y = [1 if i > 0.5 else 0  for i in y_pred]
        return y

data = np.loadtxt("trab2.data", dtype=np.float, delimiter=",")
np.random.shuffle(data)
X = data[:, :2]
y = data[:,2]
X[:,0] = normalizacao01(X[:,0])
X[:,1] = normalizacao01(X[:,1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

reg = rl(1000, 1)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(acc(y_test, y_pred))
print(cm.plot_confusion_matrix(reg, X_test, y_test))
plt.scatter(X[:,0], X[:,1], c=y,  edgecolors='k'), plt.show()
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred,  edgecolors='k'),plt.show()  