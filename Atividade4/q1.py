import kmeans as km
import numpy as np
from matplotlib import pyplot as plt
data = np.loadtxt("trab4.data", dtype=np.float, delimiter=",")
X = data.copy()


dis_mean = []
for i in range(2,6):

    temp = 0
    k = km.kmeans(X,i)
    for j in range(20):
        k.fit()
        k.show_all
        temp += k.avg_dist()
        # print(temp)
    dis_mean.append(temp/20)
# print(dis_mean)
x_a = np.arange(2,6)
print(x_a, dis_mean)
plt.plot(x_a, dis_mean)
plt.xlabel('N centroids')
plt.ylabel('distance')
plt.show()