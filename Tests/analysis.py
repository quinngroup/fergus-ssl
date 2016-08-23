import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import timeit
import sys
import os
from sklearn.cross_validation import KFold
from collections import OrderedDict
import operator
import random
var = float(sys.argv[1])
size = int(sys.argv[2])
gamma = float(sys.argv[3])
clusters = int(sys.argv[4])
dimensions = int(sys.argv[5])
#numBins = float(sys.argv[6])
dataX,dataY=datasets.make_blobs(n_samples=size, n_features=dimensions, centers=clusters, cluster_std=var, center_box=(-10.0, 10.0), shuffle=True, random_state=None)
#dataX,dataY=datasets.make_moons(n_samples=size,random_state=None)
#print dataX.shape
def labelremover(X,y):
    newX1 = np.around(X,decimals=2)
    newY1=np.copy(y)
    dim = X.shape[1]
    points = np.array(np.empty(len(np.unique(y))))
    knownX = np.empty((len(points),dim))
    knownY = np.empty(len(points))
    for i in np.unique(y):
        points[i] = np.where(y==(i))[0][0]
    for j in np.arange(0,len(newY1)):
        newY1[j]=-1
    for k in np.unique(y):
        newY1[points[k]] = y[points[k]]
    knownX = X[[i for i in points]]
    knownY = y[[i for i in points]]
    print "These are labels of known points: "+ str(knownY)
    return (newY1, knownX, knownY)

class DefaultListOrderedDict(OrderedDict):
    def __missing__(self,k):
        self[k] = []
        return self[k]

with open('/home/madhura/Computational_Olfaction/fergus-ssl/src/fergus_propagation.py') as source_file:
    exec(source_file.read())

fp = FergusPropagation(gamma = gamma)
setdict = DefaultListOrderedDict()
trainErrDict = OrderedDict()
testErrDict = OrderedDict()
knownPoint = DefaultListOrderedDict()
knownLabel = DefaultListOrderedDict()
counter=0
kf = KFold(size, n_folds=5)
for train, test in kf:
    tempList = []
    counter +=1
    trainind = train
    testind = test
    trainX = dataX[trainind]
    trainY = dataY[trainind]
    testX = dataX[testind]
    testY = dataY[testind]
    newtrainY, knownX, knownY = labelremover(trainX,trainY)
    knownPoint[counter].append(knownX)
    knownLabel[counter].append(knownY)
    '''
    scale = 2.0
    #for i in range(trainX.shape[1]):
        #sd1.append(np.std(trainX[:,i]))
    for inst in range(len(trainX)):
        for f in range(len(trainX[inst])):
                rnd = random.uniform(-trainX[inst][f], trainX[inst][f])
        trainX[inst] = trainX[inst] + float(rnd)/float(scale)
    #for i in range(testX.shape[1]):
        #sd1.append(np.std(testX[:,i]))
    for inst in range(len(testX)):
        for f in range(len(testX[inst])):
                rnd = random.uniform(-testX[inst][f], testX[inst][f])
        testX[inst] = testX[inst] + float(rnd)/float(scale)

    plt1.scatter(trainX[:,0], trainX[:,1], c=trainY, cmap = (('ocean')))
    plt1.show()
    '''
    fp.fit(trainX,newtrainY)
    #trainfunc = fp.func
    #print "train "+ str(trainfunc.shape)
    predicted_labels = fp.predict(testX)
    #testfunc = fp.func
    #print "test "+ str(testfunc.shape)
    oldTrainLabels = np.copy(trainY)
    newTrainLabels = np.copy(fp.labels_)
    TrainError = oldTrainLabels.size-np.sum((oldTrainLabels==newTrainLabels))
    oldTestLabels = np.copy(testY)
    newTestLabels = np.copy(predicted_labels)
    TestError = oldTestLabels.size-np.sum((oldTestLabels==newTestLabels))
    tempList = [trainX, trainY, testX, testY, fp.labels_, predicted_labels]
    setdict[counter].append(tempList)
    trainErrDict[counter] = TrainError
    testErrDict[counter] = TestError
    #plt.show()a

#plt.figure(0)
minInd = min(trainErrDict.iteritems(), key=operator.itemgetter(1))[0]
print "this one has been selected "+ str(minInd)
#print "this is dict: "+ str(setdict[k][0][1])
trainX = setdict[minInd][0][0]
trainY = setdict[minInd][0][1]
testX = setdict[minInd][0][2]
testY = setdict[minInd][0][3]
ori_labels = setdict[minInd][0][4]
predicted_labels = setdict[minInd][0][5]
TrainError = trainErrDict[minInd]
TestError = testErrDict[minInd]
knownX = knownPoint[minInd][0]
knownY = knownLabel[minInd][0]
labels = ['point{0}'.format(i) for i in range(len(knownY))]
plt.scatter(trainX[:, 0], trainX[:, 1], marker='o', c=trainY, cmap = ('ocean'))
plt.scatter(knownX[:,0], knownX[:,1], marker = 'D', c = knownY,cmap = (('YlOrRd')))
for label, x, y in zip(labels, knownX[:, 0], knownX[:, 1]):
    plt.annotate(label, xy = (x,y), xytext = (-20,20), textcoords = 'offset points',
    ha = 'right', va = 'bottom',bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow',
    alpha = 0.5),arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.title('Training set before labeling size: '+repr(trainX.shape[0])+' sd: '+repr(var))
#plt.scatter(trainX[labelled[:],0], trainX[labelled[:],1], marker = 'o', c=trainY[labelled[:]])
s1 = "./Results/Train_before_"+repr(trainX.shape[0])+"C"+repr(clusters)+"D"+repr(dimensions)+'k'+repr(counter)
#s1 = "./Results/Train_before_"+repr(trainX.shape[0])+"D"+repr(dimensions)+'k'+repr(counter)
plt.savefig(s1)
plt.cla()
plt.clf()
#plt.figure(1)

plt.scatter(trainX[:, 0], trainX[:, 1], marker='o', c=ori_labels, cmap = ('ocean'))
plt.scatter(knownX[:,0], knownX[:,1], marker = 'D', c = knownY, cmap = (('YlOrRd')))
for label, x, y in zip(labels, knownX[:, 0], knownX[:, 1]):
    plt.annotate(label, xy = (x,y), xytext = (-20,20), textcoords = 'offset points',
    ha = 'right', va = 'bottom',bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow',
    alpha = 0.5),arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


plt.title('Training set after labeling size: '+repr(trainX.shape[0])+ 'Train Error: '+ repr(TrainError)+' sd: '+repr(var))
s2 = "./Results/Train_after_"+repr(trainX.shape[0])+"C"+repr(clusters)+"D"+repr(dimensions)+'k'+repr(counter)
plt.savefig(s2)
plt.cla()
plt.clf()
#plt.show()
#plt.figure(0)
plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=testY, cmap = ('ocean'))
plt.title('Test set before prediction size: '+repr(testX.shape[0])+' sd: '+repr(var))
s3 = "./Results/Test_before_"+repr(testX.shape[0])+"C"+repr(clusters)+"D"+repr(dimensions)+"Bins"+'k'+repr(counter)
plt.savefig(s3)
plt.cla()
plt.clf()
#plt.figure(1)
plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=predicted_labels, cmap = ('ocean'))
plt.title('Test set after prediction size: '+repr(testX.shape[0])+'Test Error: '+ repr(TestError)+' sd: '+repr(var))
s4 = "./Results/Test_after_"+repr(testX.shape[0])+"C"+repr(clusters)+"D"+repr(dimensions)+"Bins"+'k'+repr(counter)
plt.savefig(s4)
plt.cla()
plt.clf()

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
