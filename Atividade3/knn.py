import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

data = np.loadtxt("trab3.data", dtype=np.float, delimiter=",")
np.random.shuffle(data)
X = data[:, :2]
y = data[:,2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def acc(y_real, y_pred):
    n = y_real.shape[0]
    n_true = np.count_nonzero(y_real == y_pred)
    return n_true/n 


class KNN:

    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self,x):
        dists = cdist(self.X, x.reshape((1,2)), metric='euclidean')
        idx = np.argsort(dists.reshape((dists.shape[0])))[:self.k]
        comp_labels = self.y[idx]
        # print(comp_labels)
        values,counts = np.unique(comp_labels,return_counts=True)
        ind=np.argmax(counts)
        # print (values[ind]) 
        return values[ind]

knn = KNN(5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(acc(y_test, y_pred))
