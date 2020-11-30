import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
# import cv2
import conf_matrix as cm
import plot_boundaries as pb


data = np.loadtxt("trab2.data", dtype=np.float, delimiter=",")
np.random.shuffle(data)
X = data[:, :2]
y = data[:,2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


def acc(y_real, y_pred):
    n = y_real.shape[0]
    n_true = np.count_nonzero(y_real == y_pred)
    return n_true/n    


class DQGauss:

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.pcs = {}
        self.ucs={}
        self.sigma_c={}
        n_samples, _ = X.shape

   
        for c in self.classes:
            mask = (y == c) 
            X_c = X[mask]
            # print(X_c)
            self.pcs[c] = (X_c.shape[0]/n_samples)
            self.ucs[c] = (np.mean(X_c, axis=0))
            self.sigma_c[c] = (np.cov(X_c.T))

    def predict(self, X):
        _, n_features = X.shape
        y_pred = []
        for xi in X:
            ps = []
            for c in self.classes:
                sigma_c = self.sigma_c[c]
                uc = self.ucs[c]
                pc = self.pcs[c]                
                det = np.linalg.det(sigma_c)
                f = 1/(np.sqrt(det*(2*np.pi)**n_features))
                exp = np.exp(-(1/2)*(xi-uc).T @ np.linalg.pinv(sigma_c) @ (xi-uc))
                pxc = f*exp                
                pcx = pxc*pc
                ps.append(pcx)
            y_pred.append(np.argmax(ps))
        return np.array(y_pred)

def test_DQG():
    data = np.loadtxt("trab2.data", dtype=np.float, delimiter=",")
    np.random.shuffle(data)
    X = data[:, :2]
    y = data[:,2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = DQGauss()
    # pb.plot_boundaries(clf, X, y)
    clf.fit(X_train, y_train)        
    y_pred_test = clf.predict(X_test)
    # print(y_pred_test)
    print("acc = ", acc(y_test, y_pred_test))
    print(cm.plot_confusion_matrix(clf, X_test, y_test))
    plt.scatter(X[:,0], X[:,1], c=y,  edgecolors='k'), plt.show()
    plt.scatter(X_test[:,0], X_test[:,1], c=y_pred_test,  edgecolors='k'),plt.show()  
test_DQG()



