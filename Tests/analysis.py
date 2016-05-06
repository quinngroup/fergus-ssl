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
dimensions = float(sys.argv[5])
dataX,dataY=datasets.make_blobs(n_samples=size, n_features=dimensions, centers=clusters, cluster_std=var, center_box=(-10.0, 10.0), shuffle=True, random_state=None)
plt.scatter(dataX[:, 0], dataX[:, 1], marker='o', c=dataY, cmap = ('rainbow_r'))
plt.title('size: '+repr(size)+' gamma: '+repr(gamma)+' sd: '+repr(var))
plt.show()
trainsize = (size*60)/100
randind = random.sample(range(size), int(trainsize))
leftindices = size - len(randind)
trainX = dataX[randind]
trainY = dataY[randind]
testindices = np.setdiff1d(np.arange(size),randind)
testX = dataX[testindices]
testY = dataY[testindices]
plt.figure(0)
plt.scatter(trainX[:, 0], trainX[:, 1], marker='o', c=trainY, cmap = ('rainbow_r'))
plt.figure(1)
plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=testY, cmap = ('rainbow_r'))
plt.show()

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
    points = np.empty(len(np.unique(y)))
    for i in np.unique(y):
        points[i] = np.where(y==(i))[0][0]
    for j in np.arange(0,len(newY1)):
        newY1[j]=-1
    for k in np.unique(y):
        newY1[points[k]] = y[points[k]]
    return newY1

newtrainY = labelremover(train,trainY)
with open('/home/madhura/Computational_Olfaction/fergus-ssl/src/fergus_propagation.py') as source_file:
    exec(source_file.read())

fp = FergusPropagation()
fp.fit(trainX,newtrainY)
predicted_labels = fp.predict(testX)
plt.figure(0)
plt.scatter(trainX[:, 0], trainX[:, 1], marker='o', c=trainY, cmap = ('Paired'))
plt.figure(1)
plt.scatter(trainX[:, 0], trainX[:, 1], marker='o', c=fp.labels_, cmap = ('Paired'))
plt.show()
plt.figure(0)
plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=testY, cmap = ('Paired'))
plt.figure(1)
plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=predicted_labels, cmap = ('Paired'))

'''--------------------------------------------------------------------------------------------------------------'''

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
