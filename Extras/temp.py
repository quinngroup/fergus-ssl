import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import sys
from LabelPropagationDistributed import LabelPropagationDistributed as LPD
dataX,dataY=datasets.make_blobs(n_samples=10000, n_features=50, centers=2, cluster_std=1.5, center_box=(-10.0, 10.0), shuffle=True, random_state=None)

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

trainX = dataX[0:8000,:]
trainY = dataY[0:8000]
testX = dataX[8000:10000,:]
testY = dataY[8000:10000]

newtrainY, knownX, knownY = labelremover(trainX,trainY)
file = open("stats.txt", "a+")
bins = [49,3,8,12]
k = [49,3,8,12]
trainGT = sc.parallelize(trainY)
dX = sc.parallelize(trainX)
dy = sc.parallelize(newtrainY)
testGT = sc.parallelize(testY)
for i in range(len(bins)):
    lpd = LPD(sc=sc, sqlContext = sqlContext, numBins = bins[i], k = k[i])
    lpd.fit(dX,dy)
    plabels_ = lpd.predict(sc.parallelize(testX))
    testGTAndPredicted = (testGT.zipWithIndex().map(lambda x: (x[1],x[0]))).join(plabels_.zipWithIndex().map(lambda x: (x[1],x[0]))).map(lambda (a,(b,c)): (b,c))
    tp = float(testGTAndPredicted.filter(lambda (a,b): b==1 and a==b).count())
    tn = float(testGTAndPredicted.filter(lambda (a,b): b==0 and a==b).count())
    fp = float(testGTAndPredicted.filter(lambda (a,b): b==1 and a!=b).count())
    fn = float(testGTAndPredicted.filter(lambda (a,b): b==0 and a!=b).count())
    try:
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
        f1 = 2.0*((precision*recall)/(precision+recall))
        file.write("============k: "+str(k[i])+" bins: "+str(bins[i])+"=============\n")
        file.write("precision is: "+ str(precision)+"\n")
        file.write("recall is: " + str(recall)+"\n")
        file.write("f1 measure: " + str(f1)+"\n")
    except:
        continue

file.close()
#===============================================================
dataX,dataY=datasets.make_blobs(n_samples=10, n_features=5, centers=2, cluster_std=1.5, center_box=(-10.0, 10.0), shuffle=True, random_state=None)
trainX = dataX[0:7,:]
trainY = dataY[0:7]
testX = dataX[7:10,:]
testY = dataY[7:10]
newtrainY, knownX, knownY = labelremover(trainX,trainY)
lpd = LPD(sc=sc, sqlContext = sqlContext)
X = sc.parallelize(trainX)
y = sc.parallelize(newtrainY)
lpd.fit(X,y)
#Data before PCA
plt.scatter(trainX[:,0], trainX[:,1])
plt.scatter(knownX[:,0], knownX[:,1], c=knownY,cmap=(('YlGn')) )
plt.show()
n,classes = lpd.getParams(X,y)
rotatedData = lpd.rotate(X)
#Data after PCA
rd = np.array(rotatedData.map(lambda vec: vec.toArray()).collect())
plt.scatter(rd[:,0], rd[:,1])
plt.scatter(knownX[:,0], knownX[:,1], c=knownY,cmap=(('YlGn')) )
plt.show()

#angles between rotated components
import math

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

print "for original data:"
for i in range(lpd.dimensions-1):
    print "angle between "+str(i)+" and "+ str(i+1)+" is: " + str(angle(trainX[:,i],trainX[:,i+1]) * (180/math.pi))

print "for rotated data:"
for i in range(lpd.dimensions-1):
    print "angle between "+str(i)+" and "+ str(i+1)+" is: " + str(angle(rd[:,i],rd[:,i+1]) * (180/math.pi))

#Histograms for all dimensions
dictData = lpd.makeDF(rotatedData, lpd.dimensions)
newsig,lpd.newg,lpd.newEdgeMeans = lpd.getKSmallest(dictData)
bc_EdgeMeans, bc_newg, kb = lpd.broadcaster()
dataBounds = lpd.getdataboundaries(dictData, lpd.k)
makeItMatrix = RowMatrix(dictData.rdd.map(lambda row: selectValues(row.asDict(), kb).values()))
approxValues = makeItMatrix.rows.map(lambda rw: transformer(rw, dataBounds, bc_EdgeMeans, bc_newg))
lpd.alpha = lpd.getAlpha(approxValues, y, n, newsig)
efunctions = lpd.solver(approxValues)

for i in range(lpd.dimensions):
    histograms, binEdges = lpd.approximateDensities(str(i+1), dictData)
    plt.hist(rd[:,i], bins=lpd.numBins)
    plt.show()
plabels_ = lpd.predict(sc.parallelize(testX))
#plotting
plt.scatter(trainX[:, 0], trainX[:, 1], marker='o', c=trainY, cmap = ('ocean'))
plt.show()
plt.scatter(trainX[:, 0], trainX[:, 1], marker='o', c=lpd.labels_.collect(), cmap = ('ocean'))
plt.show()
plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=testY, cmap = ('ocean'))
plt.show()
plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=plabels_.collect(), cmap = ('ocean'))
plt.show()
