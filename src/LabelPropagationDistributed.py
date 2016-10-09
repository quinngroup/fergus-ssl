# coding: utf-8
import numpy as np

from numpy.core.numeric import array

from sklearn.metrics import pairwise

from collections import OrderedDict

from scipy.linalg import eig

from scipy import interpolate as ip

from pyspark.mllib.linalg.distributed import IndexedRow, RowMatrix, IndexedRowMatrix, CoordinateMatrix, MatrixEntry
from pyspark.mllib.linalg import DenseVector, Vectors
from pyspark.mllib.feature import PCA as PCAmllib
from pyspark.sql.types import *
from pyspark.mllib.clustering import GaussianMixture

class LabelPropagationDistributed():
    """
    A Label Propagation semi-supervised clustering program using MLlib.

    >>> from LabelPropagationDistributed import LabelPropagationDistributed as LPD
    >>> dataX = array([ 5.76961775,  -6.55673209, 11.30752027,  -1.56316985,
                8.76722337,  -1.54995049, 10.23511359,  -6.20912033,
                3.49161828,  -3.02917744]).reshape(5,2)
    >>> dataY = array([ 1,  -1,  0, -1, -1])
    >>> test = array([2.1159109 ,   6.03520684,   1.04347698,  -4.44740207,
               -8.33902404,   4.20918959,   1.38447488,  -1.50363493]).reshape(4,2)
    >>> lpd = LPD(sc=sc, sqlContext = sqlContext, numBins = 5)
    >>> lpd.fit(sc.parallelize(dataX),sc.parallelize(dataY))
    >>> plabels_ = lpd.predict(sc.parallelize(test))

    """

    global transformer
    global selectValues
    global bc_EdgeMeans
    global bc_newg
    global kb

    def __init__(self, sc = None, sqlContext = None,  k = -1,numBins = -1 ,lagrangian = 10):

        self.k = k
        self.numBins = numBins
        self.lagrangian = lagrangian
        self.sqlContext = sqlContext
        self.sc = sc

    def makeDF(self,rotatedData, dimensions):
        X_ = rotatedData.map(lambda vec: np.array(vec))
        dataAsDict = X_.map(lambda x: tuple(float(f) for f in x))
        schemaString = ""
        for i in range(dimensions):
            i = i+1
            schemaString += str(i) + " "
        schemaString = schemaString.strip()
        fields = [StructField(field_name,FloatType(), True) for field_name in schemaString.split()]
        schema = StructType(fields)
        return self.sqlContext.createDataFrame(dataAsDict, schema)

    def getdataboundaries(self,dictData,k):
        dataBounds = OrderedDict()
        for i in range(0,k):
            s = str(i+1)
            tmprdd = dictData.select(s).rdd.map(lambda row: row.asDict().values()[0])
            dataBounds[i] = (tmprdd.min(),tmprdd.max())
        return dataBounds


    def transformer(vec, bounds, bc_EdgeMeans, bc_newg):
        vec1 = vec.toArray()
        tmpArr = np.zeros(vec1.shape)
        edge_means = bc_EdgeMeans.value
        for i in range(len(vec1)):
            inter2 = ip.interp1d(edge_means[:,i], bc_newg.value[:,i])
            (minVal,maxVal) = bounds.get(i)
            if (minVal < edge_means[:,i].min()) or (maxVal > edge_means[:,i].max()):
                val = (((edge_means[:,i].max()-edge_means[:,i].min())*(vec1[i] - minVal))/(maxVal - minVal)) + edge_means[:,i].min()
                if vec1[i]==minVal:
                    val = val + 0.001
                if vec1[i]==maxVal:
                    val = val - 0.001
            else:
                val = vec1[i]
            tmpArr[i] = inter2(val)
        return DenseVector(tmpArr)


    @staticmethod
    def indRowMatMaker(irm):
        return IndexedRowMatrix(irm.zipWithIndex().map(lambda x:IndexedRow(x[1],x[0])))

    def solver(self,vectors):
        U = vectors
        alpha_bc = self.sc.broadcast(self.alpha.reshape(self.k,1))
        f = U.map(lambda vec: np.dot(vec.toArray(), alpha_bc.value))
        return f

    def selectValues(ddict,kb):
        desired_keys = sorted([int(k) for k in ddict.keys()])[0:kb.value]
        newddict = { i: ddict[str(i)] for i in desired_keys }
        return newddict

    def relabel(self,labels):
        gaussians = np.zeros((self.k,1))
        i=0
        for ls in self.gmm.gaussians:
            gaussians[i] = (ls.mu)
            i = i +1
        distmeans = self.sc.broadcast(np.argsort(gaussians.flatten()))
        return labels.map(lambda x: np.where(distmeans.value == x)[0][0])

    def _get_kernel(self, X, y = None,ker=None):

        if ker == "rbf":
            if y is None:
                return pairwise.rbf_kernel(X, X, gamma = 625)
            else:
                return pairwise.rbf_kernel(X, y, gamma = 625)
        elif ker == "linear":
            if y is None:
                return pairwise.euclidean_distances(X, X)
            else:
                return pairwise.euclidean_distances(X, y)
        else:
            raise ValueError("is not a valid kernel. Only rbf and euclidean"
                             " are supported at this time")

    def getParams(self, X, y ):

        n = X.cache().count()
        self.dimensions= X.zipWithIndex().filter(lambda (ls,i): i==0).map(lambda (row,i): len(row)).collect()[0]
        classes = sorted(y.map(lambda x: (x,1)).reduceByKey(lambda a,b: a+b).map(lambda (x,arr): x).collect())
        if classes[0] == -1:
            classes = np.delete(classes, 0) # remove the -1 from this list
            if self.k == -1:
                self.k = np.size(classes)
        if self.numBins == -1:
                self.numBins = self.k + 1
        if self.k > self.dimensions:
            raise ValueError("k cannot be more than the number of features")
        return (n,classes)

    def rotate(self, X):

        XforPCA = X.map(lambda rw: Vectors.dense(rw))
        self.PCA = PCAmllib(self.dimensions).fit(XforPCA)
        rotatedData = self.PCA.transform(XforPCA)
        return rotatedData

    def approximateDensities(self, s, dictData):

        dimRdd = dictData.select(s)
        binEdges,histograms = dimRdd.rdd.map(lambda x: x.asDict().values()[0]).histogram(self.numBins)
        histograms = array(histograms)
        binEdges = np.array(binEdges)
        db = array(np.diff(binEdges),float)
        histograms = histograms/db/histograms.sum()
        histograms = histograms + 0.01
        histograms /= histograms.sum()
        return (histograms, binEdges)

    def generalizedEigenSolver(self, histograms):

        Wdis = self._get_kernel(histograms.reshape(histograms.shape[0],1),y=None,ker="linear")
        P = np.diag(histograms)
        Ddis = np.diag(np.sum((P.dot(Wdis.dot(P))),axis=0))
        Dhat = np.diag(np.sum(P.dot(Wdis),axis=0))
        sigmaVals, functions = eig((Ddis-(P.dot(Wdis.dot(P)))),(P.dot(Dhat)))
        arg = np.argsort(np.real(sigmaVals))[1]
        return (np.real(sigmaVals)[arg], np.real(functions)[:,arg])

    def getKSmallest(self, dictData):

        sig = np.zeros(self.dimensions)
        gee = np.zeros((self.numBins,self.dimensions))
        b_edgeMeans = np.zeros((self.numBins,self.dimensions))

        for i in range(self.dimensions):
            s = str(i+1)
            histograms, binEdges = self.approximateDensities(s, dictData)
            b_edgeMeans[:,i] = np.array([binEdges[j:j + 2].mean() for j in range(binEdges.shape[0] - 1)])
            sig[i], gee[:,i] = self.generalizedEigenSolver(histograms)

        if np.isnan(np.min(sig)):
            nan_num = np.isnan(sig)
            sig[nan_num] = 0
        ind =  np.argsort(sig)[0:self.k]
        return (sig[ind],gee[:,ind], b_edgeMeans[:,ind])

    def broadcaster(self):

        bc_EdgeMeans = self.sc.broadcast(self.newEdgeMeans)
        bc_newg = self.sc.broadcast(self.newg)
        kb = self.sc.broadcast(self.k)
        return (bc_EdgeMeans, bc_newg, kb)

    def getAlpha(self, approxValues, y, n, newsig):

        U = LabelPropagationDistributed.indRowMatMaker(approxValues)
        labeled_ind = np.array(y.zipWithIndex().filter(lambda (a,b): a!=-1).map(lambda (a,b): b).collect())
        matent = []
        for i in labeled_ind:
            matent.append(MatrixEntry(i,i,self.lagrangian))
        V = CoordinateMatrix(self.sc.parallelize(matent),numRows=n, numCols=n)
        Utrans = U.toCoordinateMatrix().transpose()
        Ublk = U.toBlockMatrix()
        product1 = Utrans.toBlockMatrix().multiply(V.toBlockMatrix())
        product2 = product1.multiply(Ublk)
        S = np.diag(newsig)
        localblk = product2.toLocalMatrix().toArray()
        A = S + localblk
        if np.linalg.det(A) == 0:
            A = A + np.eye(A.shape[1])*0.000001
        yblk = CoordinateMatrix(y.zipWithIndex().map(lambda x: MatrixEntry(x[1],0,x[0]))).toBlockMatrix()
        b = product1.multiply(yblk).toLocalMatrix().toArray()
        alpha = np.linalg.solve(A, b)
        return alpha

    def fit(self,X,y=None):
        """

        :params X:
          RDD of data points to be trained


        Solve for alpha

        """
        n,classes = self.getParams(X, y)
        rotatedData = self.rotate(X)
        dictData = self.makeDF(rotatedData, self.dimensions)
        newsig,self.newg,self.newEdgeMeans = self.getKSmallest(dictData)
        bc_EdgeMeans, bc_newg, kb = self.broadcaster()
        dataBounds = self.getdataboundaries(dictData, self.k)
        makeItMatrix = RowMatrix(dictData.rdd.map(lambda row: selectValues(row.asDict(), kb).values()))
        approxValues = makeItMatrix.rows.map(lambda rw: transformer(rw, dataBounds, bc_EdgeMeans, bc_newg))
        self.alpha = self.getAlpha(approxValues, y, n, newsig)
        efunctions = self.solver(approxValues)
        self.gmm = GaussianMixture.train(efunctions, np.size(classes), convergenceTol=0.0001,
                                         maxIterations=5000, seed=None)
        labels_ = self.gmm.predict(efunctions)
        self.labels_ = self.relabel(labels_)
        return self

    def predict(self, X,y=None):
        '''
        X and y should be RDDs

        '''
        bc_EdgeMeans, bc_newg, kb = self.broadcaster()
        testXforPCA = X.map(lambda rw: Vectors.dense(rw))
        newX = self.PCA.transform(testXforPCA)
        testdf = self.makeDF(newX, self.dimensions)
        testdatabounds = self.getdataboundaries(testdf, self.k)
        testmakeItMatrix = RowMatrix(testdf.rdd.map(lambda row: selectValues(row.asDict(), kb).values()))
        testapproxValues = testmakeItMatrix.rows.map(lambda rw: transformer(rw, testdatabounds,bc_EdgeMeans, bc_newg))
        testfunctions = self.solver(testapproxValues)
        predictedlabels = self.relabel(self.gmm.predict(testfunctions))
        return predictedlabels
