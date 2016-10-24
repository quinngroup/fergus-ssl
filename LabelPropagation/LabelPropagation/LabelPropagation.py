# coding: utf-8
import sys
import numpy as np
import sklearn.metrics.pairwise as pairwise
from sklearn import decomposition
from scipy import interpolate as ip
from scipy.linalg import eig
import sklearn.mixture as mixture

class LabelPropagation():
    """
    A Label Propagation semi-supervised clustering algorithm based
    on the paper "Semi-supervised Learning in Gigantic Image Collections"
    by Fergus, Weiss and Torralba.
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
    >>> from LabelPropagation import LabelPropagation as LP
    >>> dataX = np.array([[1,1], [2,3], [3,1], [4,10], [5,12], [6,13]])
    >>> dataY = np.array([0,0,0,1,1,1])
    >>> newdataY = np.array([0,-1,-1,-1,-1,1])
    >>> testX = np.array([[1,-1], [3,-0.5],[7,5]])
    >>> testY = np.array([0,0,1])
    >>> lp = LP(numBins = 5)
    >>> lp.fit(dataX,newdataY)
    >>> plabels_ = lp.predict(testX)
    """

    def __init__(self, kernel = None, k = -1,numBins = -1 ,lagrangian = 10):

        self.k = k
        self.numBins = numBins
        self.lagrangian = lagrangian

    def getParams(self,X,y):
        """
        Get all necessary parameters like total number of dimensions,
        desired number of dimensions to work on k,
        number of classes based on the labeled data available,
        number of histogram bins and total data points n.

        :param: X
            Data points

        :param: y
            Labels
        """

        classes = np.sort(np.unique(y))
        n = np.size(y)
        self.dimensions = X.shape[1]
        if classes[0] == -1:
            classes = np.delete(classes, 0) # remove the -1 from this list
            if self.k == -1:
                self.k = np.size(classes)
        if self.numBins == -1:
            self.numBins = self.k + 1
        if self.k > self.dimensions:
            raise ValueError("k cannot be more than the number of features")
        return (n,classes)



    def rotate(self,X):
        """
        Rotate the data to get independent dimensions.
        :param: X
            Data points to be rotated
        """

        try:
            self.pca= decomposition.PCA(n_components=self.dimensions)
            rotatedData = self.pca.fit_transform(X)
            return rotatedData
        except:
            print("PCA components should not be less than min(samples, n_features) and more that max(samples, n_features)",sys.exc_info()[0])
            raise

    def approximateDensities(self,i, X_):
        """
        Discretize data into bins. Returns the new histograms and corresponding bin edges.
        :param: i:
            index to select dimension of the data
        :param: X_:
            Original data
        """

        histograms,binEdges = np.histogram(X_[:,i],bins=self.numBins,density=True)
        histograms = histograms+ 0.01
        histograms /= histograms.sum()
        return(histograms,binEdges)

    def generalizedEigenSolver(self,histograms):
        """
        A generalized Eigen Solver that gives approximate eigenfunctions and eigenvalues.
        Based on Eqn. 2 in the paper.
        NOTE: The first eigenfunction is always going to be a trivial function with maximal smoothness.
              Hence selecting eigenvalue at index 1 instead of 0.

        :params: histograms:
            Discretized data whose eigenfunctions and eigenvalues are to be evaluated.
        """

        Wdis = self._get_kernel(histograms.reshape(histograms.shape[0],1),y=None,ker="linear")
        P  = np.diag(histograms)
        Ddis = np.diag(np.sum((P.dot(Wdis.dot(P))),axis=0))
        Dhat = np.diag(np.sum(P.dot(Wdis),axis=0))
        sigmaVals, functions = eig((Ddis-(P.dot(Wdis.dot(P)))),(P.dot(Dhat)))
        arg = np.argsort(np.real(sigmaVals))[1]
        return (np.real(sigmaVals)[arg], np.real(functions)[:,arg])

    def getKSmallest(self,X_):
        """
        Order and select k eigenvectors from k smallest eigenvalues.
        :param: X_:
            Data points with independent components
        """

        sig = np.zeros(self.dimensions)
        gee = np.zeros((self.numBins,self.dimensions))
        b_edgeMeans = np.zeros((self.numBins,self.dimensions))
        #self.interpolators = []

        for i in range(self.dimensions):
            histograms, binEdges = self.approximateDensities(i, X_)
            b_edgeMeans[:,i] = np.array([binEdges[j:j + 2].mean() for j in range(binEdges.shape[0] - 1)])
            sig[i],gee[:,i] = self.generalizedEigenSolver(histograms)


        if np.isnan(np.min(sig)):
            nan_num = np.isnan(sig)
            sig[nan_num] = 0

        ind =  np.argsort(sig)[0:self.k]
        return (sig[ind],gee[:,ind], b_edgeMeans[:,ind])

    def get_transformed_data(self,ori_data,edge_means,i):
        """
        Transform data points compatible with interpolator boundaries.

        :params: ori_data:
            Original data points

        :params: edge_means:
            Fit original data to boundaries of the histogram bins edges
        """

        dim = edge_means.shape[1]
        transformeddata = np.empty((ori_data.shape[0],dim))
        if ori_data[:,i].min() < edge_means[:,i].min() or ori_data[:,i].max() > edge_means[:,i].max():
            ls=[]
            for num in ori_data[:,i]:
                val = (((edge_means[:,i].max()-edge_means[:,i].min())*(num - ori_data[:,i].min()))/(ori_data[:,i].max() - ori_data[:,i].min())) + edge_means[:,i].min()
                if num==ori_data[:,i].min():
                    val = val + 0.001
                if num==ori_data[:,i].max():
                    val = val - 0.001
                ls.append(val)
            return np.array(ls)
        else:
            return ori_data[:,i]

    def transformer(self,rotatedData):
        """
        Interpolate and return approximate Eigenvectors

        :param: rotatedData:
            Data to be transformed to fit into the boundaries of the interpolator
        """

        self.interpolators = []
        transformeddata = np.zeros((rotatedData.shape[0],self.k))
        approxValues = np.zeros((rotatedData.shape[0],self.k))
        for i in range(0,self.k):
            self.interpolators.append(ip.interp1d(self.newEdgeMeans[:,i], self.newg[:,i]))
            transformeddata[:,i] = self.get_transformed_data(rotatedData,self.newEdgeMeans,i)
            approxValues[:,i] = self.interpolators[i](transformeddata[:,i])
        return approxValues

    def getAlpha(self,approxValues ,y ,n ,newsig):
        """
        Using the approximate eigenfunctions, solve Eqn 1 in the paper and
        solve it for alpha.

        :params: approxValues:
            Approximated eigenfunctions

        :params: y:
            Label array

        :params: n:
            Size of data

        :params: newsig:
            k smallest eigenvalues
        """

        U = approxValues
        S = np.diag(newsig)
        V = np.diag(np.zeros(n))
        labeled = np.where(y != -1)
        V[labeled, labeled] = self.lagrangian
        # Solve for alpha and use it to compute the eigenfunctions, f.
        A = S + np.dot(np.dot(U.T, V), U)
        if np.linalg.det(A) == 0:
            A = A + np.eye(A.shape[1])*0.000001
        b = np.dot(np.dot(U.T, V), y)
        alpha = np.linalg.solve(A, b)
        return alpha

    def relabel(self,labels):
        """
        Label the data points in an ordered way depending on the ascending
        order of the gaussian means.

        :param: labels:
            GMM predicted labels
        """
        means = np.argsort(self.gmm.means_.flatten())
        for i in range(0, np.size(labels)):
            labels[i] = np.where(means == labels[i])[0][0]
        return labels

    def solver(self ,vectors):
        """
        f =  U * alpha
        Get and return approximate eigenvectors from eigenfunctions using alpha.

        :param: vectors:
            Approximate eigenfunctions
        """

        U = vectors
        f = np.dot(U, self.alpha)
        f = f.reshape((f.shape[0],-1))
        return f

    def _get_kernel(self, X, y = None,ker=None):
        """
        Return pairwise affinity matrix based on the kernel specified.

        :param: X:
            Data array

        :param: y:
            Label array

        :param: ker:
            kernel specified. Currently supporting Euclidean and rbf.
        """

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
            raise ValueError("Not a valid kernel. Only rbf and knn"
                             " are supported at this time")

    def fit(self,X,y):
        """
        A fit function that returns a label propagation semi-supervised clustering model.
        :params X:
          Data points
        :params: y:
          Labels
        """
        if y is None:
            raise ValueError("y cannot be None")
        n,classes = self.getParams(X,y)
        rotatedData = self.rotate(X)
        newsig,self.newg,self.newEdgeMeans = self.getKSmallest(rotatedData)
        approxValues = self.transformer(rotatedData)
        self.alpha = self.getAlpha(approxValues, y, n, newsig)
        efunctions = self.solver(approxValues)
        self.gmm = mixture.GMM(n_components = np.size(classes),n_iter=5000, covariance_type='diag',min_covar=0.0000001)
        self.gmm.fit(efunctions)
        labels = self.gmm.predict(efunctions)
        self.labels_ = self.relabel(labels)
        return self

    def getVectors(self, newX,n):
        """
        For each new point, interpolate a single point and return its approximate eigen vector.
        This makes the algorithm inductive.

        :params: newX:
            Data point/points with independent components

        :params: n:
            size of data points
        """

        approxVec = np.zeros((n,self.k))
        allpoints = np.zeros((n,self.k))
        for i in range(self.k):
            allpoints[:,i] = self.get_transformed_data(newX[:,0:self.k], self.newEdgeMeans,i)
        for p in range(n):
            kpoints = allpoints[p]
            for d in range(0,self.k):
                val = kpoints[d]
                approxVec[p,d] = self.interpolators[d](val)
        return approxVec

    def predict(self,X,y=None):
        """
        Interpolate data and get approximate eigenvectors using alpha.
        Predict labels for the eigenvectors using GMM.

        :params: X:
            Data points whose labels are to be predicted
        :params: y:
            Ground Truth for data points. Label array
        """

        newX = self.pca.transform(X)
        n = newX.shape[0]
        approxVec = self.getVectors(newX,n)
        newfunctions = self.solver(approxVec)
        newlabels = self.gmm.predict(newfunctions)
        predictedLabels = self.relabel(newlabels)
        return predictedLabels
