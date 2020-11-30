import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import pdist,cdist


class Centroid:
    def __init__(self, coord):
        self.coord = coord
        self.points=[]

    def show_Cs(self):
        print ("Coord: ", self.coord)
    
    def show_Ps(self):
        self.show_Cs()
        for i in self.points:
            print("\t",i)

    def appendPoint(self, point):
        self.points.append(point)
    
    def reset_Points(self):
        self.points = []

    def update_coord(self):
        # old_Coord = self.coord
        if(len(self.points)  > 1 ):
            self.coord = np.mean(np.asarray(self.points),axis=0)
        # print("old:",old_Coord, " new:",self.coord)
        
  
class kmeans:
    X = 0
    n_cent = 0
    cent_list = []
    def __init__(self, X, cent=1):
        self.X = X
        self.n_cent = cent

    def showCents(self):
        for i in self.cent_list:
            i.show_Cs()

    def C_list2np(self):
        arr = []
        for i in self.cent_list:
            arr.append(i.coord)
        return np.asarray(arr).reshape(self.n_cent,self.X.shape[1])

    def show_all(self):
        for i in self.cent_list:
            i.show_Ps()
    
    def sct_points(self):
        dists = cdist(self.X,self.C_list2np())
        inds = np.argmin(dists, axis=1)
        for i in self.cent_list:
            i.reset_Points()
        for i in range(len(inds)):
            self.cent_list[inds[i]].appendPoint(self.X[i, :])


    def update_cent(self):
        for i in self.cent_list:
            i.update_coord()

    def init_C_list(self):
        self.cent_list = []
        i = 0
        while i < self.n_cent:
            linha = int(np.random.uniform(0, self.X.shape[0])) # seleciona uma linha aleatoria para ser um centroid
            # print("linha = ", linha)
            self.cent_list.append(Centroid(self.X[linha , :]))
            i += 1

    def fit(self):
        self.init_C_list()
        old_cents =  self.C_list2np()
        self.sct_points()
        self.update_cent()
        # i=1
        while not np.array_equal(old_cents, self.C_list2np()):
            old_cents =  self.C_list2np()
            self.sct_points()
            self.update_cent()

    def avg_dist(self):
        sum_dists = 0
        for i in self.cent_list:
            dists = cdist(np.asarray(i.coord).reshape((1,5)), np.asarray(i.points))
            sum_dists += np.sum(dists)
        return sum_dists/self.X.shape[0]

