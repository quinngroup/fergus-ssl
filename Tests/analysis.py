import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import timeit
import sys
import os
from sklearn.cross_validation import KFold

var = float(sys.argv[1])
size = int(sys.argv[2])
gamma = float(sys.argv[3])
clusters = float(sys.argv[4])
dimensions = float(sys.argv[5])
numBins = float(sys.argv[6])
dataX,dataY=datasets.make_blobs(n_samples=size, n_features=dimensions, centers=clusters, cluster_std=var, center_box=(-10.0, 10.0), shuffle=True, random_state=None)

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
    return (newY1, points)


with open('/home/madhura/Computational_Olfaction/fergus-ssl/src/fergus_propagation.py') as source_file:
    exec(source_file.read())

fp = FergusPropagation(numBins = numBins)

counter=0
kf = KFold(size, n_folds=5)
for train, test in kf:
    counter +=1
    trainind = train
    testind = test
    trainX = dataX[trainind]
    trainY = dataY[trainind]
    testX = dataX[testind]
    testY = dataY[testind]
    newtrainY, indices = labelremover(trainX,trainY)
    print ("These are indices of labeled daa: "+ repr(indices))
    fp.fit(trainX,newtrainY)
    predicted_labels = fp.predict(testX)
    oldTrainLabels = np.copy(trainY)
    newTrainLabels = np.copy(fp.labels_)
    TrainError = oldTrainLabels.size-np.sum((oldTrainLabels==newTrainLabels))
    oldTestLabels = np.copy(testY)
    newTestLabels = np.copy(predicted_labels)
    TestError = oldTestLabels.size-np.sum((oldTestLabels==newTestLabels))
    #plt.figure(0)
    plt.scatter(trainX[:, 0], trainX[:, 1], marker='o', c=trainY, cmap = ('ocean'))
    plt.title('Training set before labeling size: '+repr(trainX.shape[0])+' sd: '+repr(var))
    s1 = "./Results/Train_before_"+repr(trainX.shape[0])+"C"+repr(clusters)+"D"+repr(dimensions)+"Bins"+repr(numBins)+'k'+repr(counter)
    plt.savefig(s1)
    plt.cla()
    plt.clf()
    #plt.figure(1)
    plt.scatter(trainX[:, 0], trainX[:, 1], marker='o', c=fp.labels_, cmap = ('ocean'))
    plt.title('Training set after labeling size: '+repr(trainX.shape[0])+ 'Train Error: '+ repr(TrainError)+' sd: '+repr(var))
    s2 = "./Results/Train_after_"+repr(trainX.shape[0])+"C"+repr(clusters)+"D"+repr(dimensions)+"Bins"+repr(numBins)+'k'+repr(counter)
    plt.savefig(s2)
    plt.cla()
    plt.clf()
    #plt.show()
    #plt.figure(0)
    plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=testY, cmap = ('ocean'))
    plt.title('Test set before prediction size: '+repr(testX.shape[0])+' sd: '+repr(var))
    s3 = "./Results/Test_before_"+repr(testX.shape[0])+"C"+repr(clusters)+"D"+repr(dimensions)+"Bins"+repr(numBins)+'k'+repr(counter)
    plt.savefig(s3)
    plt.cla()
    plt.clf()
    #plt.figure(1)
    plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=predicted_labels, cmap = ('ocean'))
    plt.title('Test set after prediction size: '+repr(testX.shape[0])+'Test Error: '+ repr(TestError)+' sd: '+repr(var))
    s4 = "./Results/Test_after_"+repr(testX.shape[0])+"C"+repr(clusters)+"D"+repr(dimensions)+"Bins"+repr(numBins)+'k'+repr(counter)
    plt.savefig(s4)
    plt.cla()
    plt.clf()
    #plt.show()
'''
s2 = '/home/madhura/Computational_Olfaction/fergus-ssl/Results/' + repr(size) +'_blobs_prediction_after'
i = 0
while os.path.exists('{}{:d}.png'.format(s2, i)):
    i += 1
plt.savefig('{}{:d}.png'.format(s2, i))
plt.cla()
plt.clf()
#plt.show()
'''
