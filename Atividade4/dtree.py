import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
from scipy.stats import zscore


def gini(y):
    N = y.shape[0]
    _, counts = np.unique(y, return_counts=True)
    return 1 - np.sum([(i/N)**2 for i in counts])

class Node:
    def __init__(self, feature = None, thresh=None, left=None, right=None,*,value=None):
        # self.data = data
        self.feature = feature 
        self.thresh = thresh
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DTREE:
    def __init__(self): 
        self.root = None

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _grow_tree(self, X, y ):
        n_samp, n_feat = X.shape
        cnt_labels = len(np.unique(y))  

        if(cnt_labels == 1 or n_samp <=2):
            leaf_val = self._most_common_label(y)
            return Node(value=leaf_val)
        
        feat_idxs = np.random.choice(n_feat, n_feat,replace=False)
        best_feat, best_tresh = self._best_question(X, y, feat_idxs)
        left_idx, right_idx = self._split(X[:,best_feat],best_tresh)

        left = self._grow_tree(X[left_idx,:],y[left_idx])
        right = self._grow_tree(X[right_idx,:], y[right_idx])
        return Node(best_feat, best_tresh, left, right)

    def _split(self, X_col, split_thresh):
        left_idx = np.argwhere(X_col <= split_thresh).flatten()
        right_idx = np.argwhere(X_col > split_thresh).flatten()
        return left_idx, right_idx

    def info_gain(self, X_col, y, thresh):
        n = len(y)
        g_parent = gini(y)
        l_idx, r_idx = self._split(X_col, thresh)
        n_l, n_r = len(l_idx), len(r_idx) 
        g_left = gini(y[l_idx])
        g_right = gini(y[r_idx])
        if(len(l_idx) == 0 or len(r_idx) == 0):
            return 0
        # g_total = g_parent - ((n_l/n)*g_left + (n_r/n)*g_right)
        return g_parent - ((n_l/n)*g_left + (n_r/n)*g_right)

    def _best_question(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        # print("feat_idxs = ", feat_idxs)
        for i in feat_idxs:
            X_col = X[:,i]
            thresh = np.unique(X_col)
            for t in thresh:
                gain = self.info_gain(X_col, y, t)
                if(gain > best_gain):
                    best_gain = gain 
                    split_idx = i
                    split_thresh = t
        return split_idx, split_thresh

    def fit(self, X, y):
        self.root = self._grow_tree(X,y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        # print()
        if x[node.feature] <= node.thresh:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)



# dt = DTREE(X,y)
# dt._grow_tree
# print(gini(y))