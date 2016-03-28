import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import timeit
import sys
import os

var = float(sys.argv[1])
size = int(sys.argv[2])
gamma = float(sys.argv[3])
X1,Y1=datasets.make_blobs(n_samples=size, n_features=2, centers=2, cluster_std=var, center_box=(-10.0, 10.0), shuffle=True, random_state=None)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, cmap = ('Paired'))
s1 = '/home/madhura/Computational_Olfaction/fergus-ssl/Results/' + repr(size) +'_blobs_before_prediction'
plt.savefig(s1)
#plt.show()
newX1 = np.around(X1,decimals=2)
newY1=np.copy(Y1)
point1 = np.where(Y1==(0))[0][0]
point2 = np.where(Y1==(1))[0][0]
for y in range(0,len(newY1)):
    newY1[y]=-1

newY1[point1] = Y1[point1]
newY1[point2] = Y1[point2]
sys.argv=[gamma]
execfile('/home/madhura/Computational_Olfaction/fergus-ssl/src/fergus_propagation.py')
fp = FergusPropagation()
start = timeit.default_timer()
fp.fit(X1,newY1)
stop = timeit.default_timer()
print ("Results for blob size " + str(size)+" and standard deviation "+str(var))
print ("Runtime: "+ str(stop - start))
print ("Gamma is "+ str(fp.gamma))
print ("blob variance is "+ str(var))
print ("data size is" + str(size))
#plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, cmap = ('Paired'))
#plt.show()
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=fp.labels_, cmap = ('Paired'))
plt.title('size: '+repr(size)+' gamma: '+repr(gamma)+' sd: '+repr(var))
s2 = '/home/madhura/Computational_Olfaction/fergus-ssl/Results/' + repr(size) +'_blobs_after_prediction'
i = 0
while os.path.exists('{}{:d}.png'.format(s2, i)):
    i += 1
plt.savefig('{}{:d}.png'.format(s2, i))
#plt.show()
