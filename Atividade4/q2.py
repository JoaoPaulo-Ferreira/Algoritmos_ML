import pca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

data = np.loadtxt("trab4.data", dtype=np.float, delimiter=",")
X = data.copy()
p = pca.PCA(2)
var_preserv = p.fit(X)
new_data = p.project(X)

print("Variância Preservada: ", var_preserv)




label = X[:,4]
colors = ['#6060c0', '#487f9c', '#00b3ca']
fig = plt.figure()
plt.scatter(new_data[:,0], new_data[:,1], c=label, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()
