# coding: utf-8
# coding: utf-8
import numpy as np

from numpy.core.numeric import array

from sklearn.metrics import pairwise

from collections import OrderedDict

from scipy.linalg import eig

from scipy import interpolate as ip
from pyspark.ml.feature import PCA as pcaml
from pyspark.ml.linalg import Vectors as mlvec

from pyspark.mllib.linalg.distributed import IndexedRow, RowMatrix, IndexedRowMatrix, CoordinateMatrix, MatrixEntry
from pyspark.mllib.linalg import DenseVector, Vectors
from pyspark.mllib.feature import PCA as PCAmllib
from pyspark.sql.types import *
from pyspark.mllib.clustering import GaussianMixture
from pyspark.ml.clustering import GaussianMixture as gmmml
from pyspark.mllib.clustering import KMeans
from pyspark.ml.clustering import KMeans as kmml

class LabelPropagationDistributed():
    """
    A Label Propagation semi-supervised clustering algorithm based
    on the paper "Semi-supervised Learning in Gigantic Image Collections"
    by Fergus, Weiss and Torralba with modifications to fit Spark.
    The algorithm begins with discretizing the datapoints into 1-D histograms.
    For each independent dimension, it approximates the density using the histogram
    and solves numerically for eigenfunctions and eigenvalues.
    Then it uses the k eigenfunctions for k smallest eigenvalues to do a 1-D
    interpolation of every data point.
    These interpolated data points are the approximated eigenvectors of the Normalized
    Laplacian of the original data points which are further used to solve a k*k system
    of linear equation which yields alpha.
    The alpha is then used to get approximate functions which are then clustered using
    Gaussian Mixture Model.
    For any new/ unseen data point, the point can be interpolated and using the alpha,
    the approximate function for that point can be obtained whose label can be easily predicted
    using the GMM model learned before making it inductive learning.

    Based on
    U{https://cs.nyu.edu/~fergus/papers/fwt_ssl.pdf}
    Fergus, Weiss and Torralba, Semi-supervised Learning in Gigantic
    Image Collections, Proceedings of the 22nd International Conference
    on Neural Information Processing Systems, p.522-530, December 07-10, 2009,
    Vancouver, British Columbia, Canada

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

    def __init__(self, sc = None, sqlContext = None,  k = -1,numBins = -1 ,lagrangian = 1000, gamma = 0.01):

        self.k = k
        self.numBins = numBins
        self.lagrangian = lagrangian
        self.sqlContext = sqlContext
        self.sc = sc
        self.gamma = gamma

    @classmethod
    def print_test(cls):
        print "working"

    def makeDF(self,rotatedData, dimensions):
        """
        Convert data from RDD to a dataframe with every feature
        as a different column.

        :param rotatedData:
            Data points with as independent dimensions as possible

        :param dimensions
            Total number of dimensions of the data

        """
        X_ = rotatedData.map(lambda (ind, vec): (ind,np.array(vec)))
        dataAsDict = X_.map(lambda (ind,x): tuple([ind]+[float(f) for f in x]))
        fields = [StructField("idd",IntegerType(), True)]
        fields += [StructField(str(i),FloatType(), True) for i in range(1,dimensions+1)]
        """
        schemaString = "index "
        for i in range(1,dimensions+1):
            schemaString += str(i) + " "
        schemaString = schemaString.strip()
        fields = [StructField(field_name,FloatType(), True) for field_name in schemaString.split()]
        """
        schema = StructType(fields)
        return self.sqlContext.createDataFrame(dataAsDict, schema)


    def getdataboundaries(self,dictData,k):
        """
        For interpolating the data points, get min and max values for
        the data set.

        :param: dictData:
            Dataframe of dataset

        :param: k:
            The number of eigenvectors to be selected. Default is equal
            to the number of clusters.

        """
        dataBounds = [None for _ in range(k)]
        for i in range(0,k):
            ind = self.jj[i]
            s = str(ind + 1)
            tmp = dictData.select(s).rdd.map(lambda r: r[0]).collect()
            Lower, Upper = np.percentile(tmp, [2.5,97.5])
            dataBounds[i] = (Lower,Upper)
        return dataBounds
        
    def transformer(vec, bounds, bc_EdgeMeans, bc_newg, kb):
        """
        Interpolate and return data points
        Runs on workers

        :param: vec:
            k dimenional data

        :param: bounds:
            Min and Max values for data to be interpolated.

        :param: bc_EdgeMeans:
            Histogram bin edges to build the 1-D interpolator

        :param: bc_newg:
            k smallest eigenvectors to build the 1-D interpolator
        """
        vec1 = np.array(vec)
        tmpArr = np.zeros(vec1.shape)
        edge_means = bc_EdgeMeans.value
        for i in range(0, kb.value):
            inter2 = ip.interp1d(edge_means[:,i], bc_newg.value[:,i], fill_value = 'extrapolate', kind = 'linear')
            (lower, upper) = bounds.value[i]
            val = vec1[i]
            clippedValue = np.clip(val,lower,upper)
            tmpArr[i] = inter2(val)
        return DenseVector(tmpArr)


    @staticmethod
    def indRowMatMaker(irm):
        """
        Converts RDD to indexed Row Matrix

        :param: irm:
            RDD to be converted to IndexedRowMatrix
        """

        return IndexedRowMatrix(irm.zipWithIndex().map(lambda x:IndexedRow(x[1],x[0])))

    def solver(self,functions, alpha):
        """
        f = U * alpha
        Get and return approximate eigenvectors from eigenfunctions using alpha.

        :param: vectors:
            Approximate eigenfunctions
        """

        U = functions
        alpha_bc = self.sc.broadcast(alpha.reshape(self.k,1))
        f = U.rows.map(lambda row: (row.index, np.dot(row.vector.toArray(), alpha_bc.value)))
        return f

    def selectValues(ddict,kb):
        """
        Select and return k eigenvectors from the Dataframe.

        :param: ddict:
            Dataframe Row as a dictionary.

        :param: kb:
            Desired number of eigenvectors to be selected.
            Default equals to number of clusters

        """

        desired_keys = sorted([int(k) for k in ddict.keys()])[0:kb.value]
        newddict = { i: ddict[str(i)] for i in desired_keys }
        return newddict

    def relabel(self,ilabels, km):
        """
        Label the data points in an ordered way depending on the ascending
        order of the gaussian means.

        :param: labels:
            GMM predicted labels

        """

        #gaussians = np.zeros((self.numClasses,self.k))
        #i=0
        #for ls in self.gmm.gaussians:
        #    gaussians[i] = (ls.mu)
        #    i = i +1
        #gaussians = np.array(gmodel.gaussiansDF.select('mean').rdd.map(lambda r: r[0].toArray()).collect()).flatten()
        means = np.array(km.clusterCenters())
        distmeans = self.sc.broadcast(np.argsort(means.flatten()))
        ilabels = ilabels.select(["idd","prediction"]).rdd.map(lambda row: (row.idd, np.where(distmeans.value == row.prediction)[0][0]))
        return ilabels

    def _get_kernel(self, X, y = None,ker=None):
        """
        return pairwise affinity matrix based on the kernel specified.

        :param: X:
            Data array

        :param: y:
            Label array

        :param: ker:
            kernel specified. Currently supporting Euclidean and rbf.

        """

        if ker == "rbf":
            if y is None:
                return pairwise.rbf_kernel(X, X, gamma = self.gamma)
            else:
                return pairwise.rbf_kernel(X, y, gamma = self.gamma)
        elif ker == "linear":
            if y is None:
                return pairwise.euclidean_distances(X, X)
            else:
                return pairwise.euclidean_distances(X, y)
        else:
            raise ValueError("is not a valid kernel. Only rbf and euclidean"
                             " are supported at this time")

    def getParams(self, X, y ):
        """
        Get all necessary parameters like total number of dimensions,
        desired number of dimensions to work on k,
        number of classes based on the labeled data available,
        number of histogram bins and total data points n.

        :param: X
            RDD of index, Data points

        :param: y
            RDD of index, labels

        """

        n = X.cache().count()
        self.dimensions= len(X.take(1)[0][1])
        classes = sorted(y.filter(lambda (i,label): label != -1).map(lambda (i,label): label).distinct().collect())
        if classes[0] == -1:
            classes = np.delete(classes, 0) # remove the -1 from this list

        if self.k == -1:
            self.k = np.size(classes)
        if self.numBins == -1:
                self.numBins = self.k + 1
        if self.k > self.dimensions:
            self.k = self.dimensions
        return (n,classes)

    def rotate(self, X):
        """
        Rotate the data to get independent dimensions.

        :param: X
            RDD of data points to be rotated

        """
        XforPCA = X.map(lambda (ind,vec): (int(ind),mlvec.dense(vec))).toDF(schema = ('idd','features'))
        pca = pcaml(k=self.dimensions,inputCol="features", outputCol="pcaFeatures")
        self.PCA = pca.fit(XforPCA)
        rotatedData = self.PCA.transform(XforPCA)

        rotatedData=rotatedData.select(["idd", "pcaFeatures"]).rdd.map(lambda r: (r.idd,r.pcaFeatures.toArray()))
        return rotatedData

    def approximateDensities(self, s, dictData):
        """
        Discretize data into bins. Returns the new histograms and corresponding bin edges.

        :param: s:
            index to select dimension of the data

        :param: dictData:
            Dataframe of original data

        """
        dimRdd = dictData.select(s)
        binEdges,histograms = dimRdd.rdd.map(lambda x: x[0]).histogram(self.numBins)
        histograms = array(histograms)
        binEdges = np.array(binEdges)
        #db = array(np.diff(binEdges),float)
        #histograms = histograms/db/histograms.sum()
        histograms = histograms + 0.1
        histograms /= histograms.sum()
        return (histograms, binEdges)

    def generalizedEigenSolver(self, binmeans, histograms):
        """
        A generalized Eigen Solver that gives approximate eigenfunctions and eigenvalues.
        Based on Eqn. 2 in the paper.
        NOTE: The first eigenfunction is always going to be a trivial function with maximal smoothness.
              Hence selecting eigenvalue at index 1 instead of 0.

        :params: histograms:
            Discretized data whose eigenfunctions and eigenvalues are to be evaluated.

        """

        Wdis = self._get_kernel(binmeans.reshape(binmeans.shape[0],1),y=None,ker="rbf")
        P = np.diag(histograms).astype('float32')
        PW = np.dot(P,Wdis)
        PWP = np.dot(PW,P)
        Ddis = np.diag(np.sum(PWP,axis=0))
        Dhat = np.diag(np.sum(PW,axis=0))
        L = Ddis - PWP
        IP = np.linalg.inv(np.sqrt(P.dot(Dhat)))
        L2 = IP.dot(L).dot(IP)
        uu,ss,vv = np.linalg.svd(L2)
        g = IP.dot(uu)
        s = np.diag(g.T.dot(L).dot(g))/np.diag(g.T.dot(P).dot(Dhat).dot(g))
        return s,g

    def getKSmallest(self, dictData):
        """
        Order and select k eigenvectors from k smallest eigenvalues.

        :param: dictData:
            Dataframe of data points

        """
        eigvals = np.zeros((self.numBins, self.dimensions))
        eigvecs = np.zeros(((self.numBins, self.numBins, self.dimensions)))
        b_edgeMeans = np.zeros((self.numBins,self.dimensions))
        for i in range(self.dimensions):
            s = str(i+1)
            histograms, binEdges = self.approximateDensities(s, dictData)
            b_edgeMeans[:,i] = np.array([binEdges[j:j + 2].mean() for j in range(binEdges.shape[0] - 1)])
            eigvals[:,i], eigvecs[:,:,i] = self.generalizedEigenSolver(b_edgeMeans[:,i], histograms)

        all_evals = eigvals.flatten()
        smalls = np.where(all_evals < 1e-10)
        all_evals[smalls] = 1e10
        ind =  np.argsort(all_evals)
        k_ind = ind[0:self.k]
        self.kind = k_ind
        useful_evals = np.diag(all_evals[k_ind])
        self.ii,self.jj = np.unravel_index(k_ind, eigvals.shape)
        evectors = np.column_stack([eigvecs[:,self.ii[a],self.jj[a]] for a in range(self.k)])
        return (useful_evals, evectors, b_edgeMeans[:,self.jj])

    def broadcaster(self):
        """
        Function to broadcast parameters that will be used on workers

        """

        bc_EdgeMeans = self.sc.broadcast(self.newEdgeMeans)
        bc_newg = self.sc.broadcast(self.newg)
        kb = self.sc.broadcast(self.k)
        return (bc_EdgeMeans, bc_newg, kb)

    def getAlpha(self, approxValues, y, n, newsig):
        """
        Using the approximate eigenfunctions, solve Eqn 1 in the paper and
        solve it for alpha.

        :params: approxValues:
            Approximated eigenfunctions

        :params: y:
            RDD of label array

        :params: n:
            Size of data

        :params: newsig:
            k smallest eigenvalues

        """

        U = approxValues
        labeled_ind = np.array(y.filter(lambda (a,b): b!=-1).map(lambda (a,b): a).collect())
        matent = []
        for i in labeled_ind:
            matent.append(MatrixEntry(i,i,self.lagrangian))
        V = CoordinateMatrix(self.sc.parallelize(matent),numRows=n, numCols=n)
        Utrans = U.toCoordinateMatrix().transpose()
        Ublk = U.toBlockMatrix()
        product1 = Utrans.toBlockMatrix().multiply(V.toBlockMatrix())
        product2 = product1.multiply(Ublk)
        S = newsig
        localblk = product2.toLocalMatrix().toArray()
        A = S + localblk
        if np.linalg.det(A) == 0:
            A = A + np.eye(A.shape[1])*0.000001
        yblk = CoordinateMatrix(y.map(lambda x: MatrixEntry(x[0],0,x[1]+1 if x[1] > -1 else 0))).toBlockMatrix()
        b = product1.multiply(yblk).toLocalMatrix().toArray()
        alpha = np.linalg.solve(A, b)
        vy = y.map(lambda (i,v): ((v+1)*1000) if v != -1 else 0)
        uvy = U.rows.zipWithIndex().map(lambda x: (x[1],x[0])).join(vy.zipWithIndex().map(lambda x: (x[1],x[0]))).map(lambda (a,(b,c)): IndexedRow(b.index,np.array(b.vector)*c)).flatMap(lambda irow: [(i,x) for i,x in enumerate(irow.vector.toArray())]).reduceByKey(lambda a,b: a+b).map(lambda (a,b): (a,b)).collect()
        bdash = np.array(sorted(uvy, key = lambda x: x[0]))[:,1].reshape(self.k,1)
        yi = self.sc.broadcast(OrderedDict(y.filter(lambda (i,val): val !=-1).collect()))
        lg = self.sc.broadcast(self.lagrangian)
        def multiplier(irow, yi, lg):
            if irow.index in yi.value.keys():
                return np.array(irow.vector)* lg.value
            else:
                return np.array(irow.vector)*0.0
        ilabels = self.sc.broadcast(y.filter(lambda (i,val): val !=-1).map(lambda x: x[0]).collect())
        VU = self.sc.broadcast(OrderedDict(U.rows.map(lambda irow: (int(irow.index),multiplier(irow, yi, lg))).filter(lambda rw: rw[0] in ilabels.value).collect()))
        def productMaker(irow, VU):
            ls = []
            if irow.index in VU.keys():
                for i in irow.vector:
                    for j in VU[int(irow.index)]:
                        ls.append(i*j)

                return np.array(ls).reshape(len(irow.vector),len(irow.vector))
            else:
                return 'N'

        Adash = S + np.array(U.rows.map(lambda irow: productMaker(irow, VU.value)).filter(lambda r: r!='N').map(lambda arr: (1,arr)).reduceByKey(lambda a,b: a + b).flatMap(lambda (a,vec): vec).collect())
        alpha = np.linalg.solve(Adash,bdash)
        return alpha.reshape(self.k,1)

    def fit(self,X,y):
        """
        A fit function that returns a label propagation semi-supervised clustering model.

        :params X:
          RDD of (index, data points)

        :params: y:
          RDD of (index, labels)

        """

        if y is None:
            raise ValueError("y cannot be None")


        self.n,classes = self.getParams(X, y)
        self.numClasses = np.size(classes)
        labeledPoints = self.sc.broadcast(y.filter(lambda (i,v): v!=-1).map(lambda (i,v): i).collect())
        self.labeledX = X.filter(lambda (i,vec): i in labeledPoints.value).map(lambda (a,b): b)
        self.labeledy = y.filter(lambda (i,v): v!=-1).map(lambda (a,b): b)
        rotatedData = self.rotate(X)
        dictData = self.makeDF(rotatedData, self.dimensions)
        self.newsig, self.newg, self.newEdgeMeans = self.getKSmallest(dictData)
        bc_EdgeMeans, bc_newg, kb = self.broadcaster()
        dataBounds = self.sc.broadcast(self.getdataboundaries(dictData, self.k))
        useful_data = dictData.select(['idd']+[str(i+1) for i in self.jj])
        jj = self.sc.broadcast(self.jj)
        useful_dataRDD = useful_data.rdd.map(lambda rw: (int(rw['idd']), [rw[str(i+1)] for i in jj.value]))
        approxValues = IndexedRowMatrix(useful_dataRDD.map(lambda (ind,vec) : IndexedRow(ind, transformer(vec, dataBounds, bc_EdgeMeans, bc_newg, kb)) ))
        sumofSquares = approxValues.rows.flatMap(lambda  irow: [(i,x) for i,x in enumerate(irow.vector.toArray()**2)]).reduceByKey(lambda a,b: a+b).map(lambda (a,b): (a,b)).collect()
        sos = self.sc.broadcast(np.array(sorted(sumofSquares, key = lambda x: x[0]))[:,1])
        norm_approxValues = IndexedRowMatrix(approxValues.rows.map(lambda irow: IndexedRow(irow.index,np.divide(np.array(irow.vector, dtype = 'float32'), np.array(sos.value, dtype = 'float32')))))
        self.interpolated = norm_approxValues
        self.alpha = self.getAlpha(norm_approxValues, y, self.n, self.newsig)
        efunctions = self.solver(norm_approxValues, self.alpha)
        efunctions = efunctions.map(lambda (ind,val):(ind,mlvec.dense([val]))).toDF(schema = ("idd","features"))
        #self.gmm = gmmml(featuresCol="features", predictionCol="prediction", k=np.size(classes), probabilityCol="probability", tol=0.0000001, maxIter=5000)
        #self.gmodel = self.gmm.fit(efunctions)
        #self.efunctions = efunctions
        #labeled=self.gmodel.transform(efunctions)
        self.kmeans = kmml(featuresCol="features",predictionCol="prediction", k= np.size(classes), initMode="k-means||",initSteps=10, tol=1e-4, maxIter=2000, seed=0)
        self.kmodel = self.kmeans.fit(efunctions)
        labeled=self.kmodel.transform(efunctions)
        self.labels_ = self.relabel(labeled, self.kmodel)
        return self

    def predict(self, X,y=None):
        """
        Interpolate data and get approximate eigenvectors using alpha.
        Predict labels for the eigenvectors using GMM.

        :params: X:
            RDD of data points whose label is to be predicted

        :params: y:
            Ground Truth for data points. RDD of Label array

        """
        oldn = X.count()
        minusOnes = self.sc.parallelize(np.array([-1] * oldn))
        y =  (minusOnes + self.labeledy).zipWithIndex().map(lambda (a,b): (b,a))
        lpoints = self.labeledy.count()
        X = (X.map(lambda (i,v): v) + self.labeledX).zipWithIndex().map(lambda (a,b): (b,a))
        newn = X.count()
        XforPCA = X.map(lambda (ind,vec): (int(ind),mlvec.dense(vec))).toDF(schema = ('idd','features'))
        rotatedData = self.PCA.transform(XforPCA)
        rotatedData=rotatedData.select(["idd", "pcaFeatures"]).rdd.map(lambda r: (r.idd,r.pcaFeatures.toArray()))
        dictData = self.makeDF(rotatedData, self.dimensions)
        bc_EdgeMeans, bc_newg, kb = self.broadcaster()
        dataBounds = self.sc.broadcast(self.getdataboundaries(dictData, self.k))
        useful_data = dictData.select(['idd']+[str(i+1) for i in self.jj])
        jj = self.sc.broadcast(self.jj)
        useful_dataRDD = useful_data.rdd.map(lambda rw: (int(rw['idd']), [rw[str(i+1)] for i in jj.value]))
        approxValues = IndexedRowMatrix(useful_dataRDD.map(lambda (ind,vec) : IndexedRow(ind, transformer(vec, dataBounds, bc_EdgeMeans, bc_newg, kb)) ))
        sumofSquares = approxValues.rows.flatMap(lambda  irow: [(i,x) for i,x in enumerate(irow.vector.toArray()**2)]).reduceByKey(lambda a,b: a+b).map(lambda (a,b): (a,b)).collect()
        sos = self.sc.broadcast(np.array(sorted(sumofSquares, key = lambda x: x[0]))[:,1])
        norm_approxValues = IndexedRowMatrix(approxValues.rows.map(lambda irow: IndexedRow(irow.index,np.divide(np.array(irow.vector, dtype = 'float32'), np.array(sos.value, dtype = 'float32')))))
        self.tinterpolated = norm_approxValues
        self.talpha = self.getAlpha(norm_approxValues, y, newn, self.newsig)
        efunctions = self.solver(norm_approxValues, self.talpha)
        efunctions = efunctions.map(lambda (ind,val):(ind,mlvec.dense([val]))).toDF(schema = ("idd","features"))
        kmeans = kmml(featuresCol="features",predictionCol="prediction", k= self.numClasses, initMode="k-means||",initSteps=10, tol=1e-4, maxIter=2000, seed=0)
        kmodel = kmeans.fit(efunctions)
        labeled=kmodel.transform(efunctions)
        predicted = self.relabel(labeled, kmodel)
        return predicted.filter(lambda (i,v): i < (newn - lpoints))
