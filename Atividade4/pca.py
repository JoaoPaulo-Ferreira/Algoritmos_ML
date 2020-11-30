import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import zscore

data = np.loadtxt("trab4.data", dtype=np.float, delimiter=",")
X = data.copy()


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        # self.mean = None

    def cov_XY(self, X, Y):
        n = len(X)
        x_mean = np.mean(X)
        y_mean = np.mean(Y)
        return 1/n * np.sum(np.dot((X - x_mean), (Y - y_mean).T))

    def cov_XX(self, X):
        n = len(X)
        x_mean = np.mean(X)
        return 1/n * np.sum(np.dot((X-x_mean),(X-x_mean).T))

    def cov_Matrix(self,X):
        n_feat = X.shape[1]
        cov_m = np.zeros((n_feat,n_feat))
        for i in range(n_feat):
            for j in range(n_feat):
                if(i != j):
                    cov_m[i,j] = self.cov_XY(X[:,i],X[:,j])
                else:
                    cov_m[i,j] = self.cov_XX(X[:,i])
        return cov_m

    def fit(self, X):
        X = zscore(X)
        cov_m = self.cov_Matrix(X)
        e_val, e_vet = np.linalg.eig(cov_m)
        e_vet = e_vet.T
        e_index = np.argsort(e_val)[::-1]
        e_val = e_val[e_index]
        e_vet = e_vet[e_index]
        self.components = e_vet[0:self.n_components]

        e_total = np.sum(e_val)
        e_acc = np.sum(e_val[0:self.n_components])
        return e_acc / e_total


    def project(self, X):
        x_mean = np.mean(X, axis=0)
        X = X - x_mean
        return np.dot(X, self.components.T)

    # def var_preserved(self):
    #     e_total =   


# p = PCA(2)
# var_presert = p.fit(X)
# print("var_presert = ", var_presert)
# print("cov(x,y) = ", p.cov_XY(X[0,:], X[1,:]))
# print("cov(x,x) = ", p.cov_XX(X[0,:]))
# print(np.cov(X.T))
# print(X[:2,:])
# print(X[0,:], X[1,:])
# mean = np.mean(X, axis=0)
# std = np.std(X, axis=0)
# J = (X - mean)/std.T
# J = zscore(X)
# print(X.shape)
# print(J.shape)
# print(J[0:6, :])
# print(X[0:6, :])