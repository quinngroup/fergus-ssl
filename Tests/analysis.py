import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import timeit
import sys
import os

var = float(sys.argv[1])
size = int(sys.argv[2])
gamma = float(sys.argv[3])
clusters = float(sys.argv[4])
X1,Y1=datasets.make_blobs(n_samples=size, n_features=2, centers=clusters, cluster_std=var, center_box=(-10.0, 10.0), shuffle=True, random_state=None)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, cmap = ('Paired'))
plt.title('size: '+repr(size)+' gamma: '+repr(gamma)+' sd: '+repr(var))
plt.axis([-20,20,-20,20])
s1 = '/home/madhura/Computational_Olfaction/fergus-ssl/Results/' + repr(size) +'_blobs_prediction_before'
i = 0
while os.path.exists('{}{:d}.png'.format(s1, i)):
    i += 1
plt.savefig('{}{:d}.png'.format(s1, i))
plt.cla()
plt.clf()
#plt.show()
def labelremover(X,y):
    newX1 = np.around(X,decimals=2)
    newY1=np.copy(y)
    dim = X.shape[1]
    points = np.empty(dim)
    for i in np.arange(dim):
        points[i] = np.where(y==(i))[0][0]
    for j in range(0,len(newY1)):
        newY1[j]=-1
    for k in np.arange(dim):
        newY1[points[k]] = y[points[k]]
    return newY1
newY1 = labelremover(X1,Y1)
nX,nY=datasets.make_blobs(n_samples=3000, n_features=2, centers=clusters, cluster_std=3.5, center_box=(-10.0, 10.0), shuffle=True, random_state=None)
plt.scatter(nX[:, 0], nX[:, 1], marker='o', c=nY, cmap = ('Paired'))
newY2 = labelremover(nX,nY)
#sys.argv=[gamma]
with open('/home/madhura/Computational_Olfaction/fergus-ssl/src/fergus_propagation.py') as source_file:
    exec(source_file.read())
#execfile('/home/madhura/Computational_Olfaction/fergus-ssl/src/fergus_propagation.py')
fp = FergusPropagation()
fp.fit(X1,newY1)
predicted_labels = fp.predict(nX,newY2)
oldLabels = np.copy(Y1)
newLabels = np.copy(fp.labels_)
diff = oldLabels.size-np.sum((oldLabels==newLabels))
print ("Number of incorrectly clustered points:" + str(diff))
print ("Results for blob size " + str(size)+" and standard deviation "+str(var))
print ("Runtime: "+ str(stop - start))
print ("Gamma is "+ str(fp.gamma))
print ("blob variance is "+ str(var))
print ("data size is" + str(size))
#plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, cmap = ('Paired'))
#plt.show()
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=fp.labels_, cmap = ('Paired'))
plt.title('size: '+repr(size)+' gamma: '+repr(gamma)+' sd: '+repr(var))
plt.axis([-20,20,-20,20])
s2 = '/home/madhura/Computational_Olfaction/fergus-ssl/Results/' + repr(size) +'_blobs_prediction_after'
i = 0
while os.path.exists('{}{:d}.png'.format(s2, i)):
    i += 1
plt.savefig('{}{:d}.png'.format(s2, i))
plt.cla()
plt.clf()
#plt.show()
