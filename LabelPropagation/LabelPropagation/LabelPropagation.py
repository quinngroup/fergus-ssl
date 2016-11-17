# coding: utf-8
import sys
import numpy as np
import sklearn.metrics.pairwise as pairwise
from sklearn import decomposition
from scipy import interpolate as ip
from scipy.linalg import eig
import sklearn.mixture as mixture
from sklearn.cluster import KMeans
class LabelPropagation():
    """
    A Label Propagation semi-supervised clustering algorithm based
    on the paper "Semi-supervised Learning in Gigantic Image Collections"
    by Fergus, Weiss and Torralba.
    The algorithm begins with discretizing the datapoints into 1-D ams.
    For each independent dimension, it approximates the density using the am
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

    def __init__(self, kernel = None, k = -1,numBins = -1 ,lagrangian = 1000, gamma = 0.2):

        self.k = k
        self.numBins = numBins
        self.lagrangian = lagrangian
        self.gamma = gamma

    def getParams(self,X,y):
        """
        Get all necessary parameters like total number of dimensions,
        desired number of dimensions to work on k,
        number of classes based on the labeled data available,
        number of am bins and total data points n.

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

        histograms,binEdges = np.histogram(X_[:,i],bins=self.numBins,density=False)
        histograms = histograms+ 0.1
        histograms /= histograms.sum()
        #print histograms
        return(histograms,binEdges)

    def generalizedEigenSolver(self,binmeans, histograms):
        """
        A generalized Eigen Solver that gives approximate eigenfunctions and eigenvalues.
        Based on Eqn. 2 in the paper.

        :params: histograms:
            Discretized data whose eigenfunctions and eigenvalues are to be evaluated.
        """

        Wdis = self._get_kernel(binmeans.reshape(binmeans.shape[0],1),y=None,ker="rbf") #
        P  = np.diag(histograms).astype('float32')
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


    def getKSmallest(self,X_):
        """
        Order and select k eigenvectors from k smallest eigenvalues.
        :param: X_:
            Data points with independent components
        """

        eigvals = np.zeros((self.numBins, self.dimensions))
        eigvecs = np.zeros(((self.numBins, self.numBins, self.dimensions)))
        b_edgeMeans = np.zeros((self.numBins,self.dimensions))

        for i in range(self.dimensions):
            histograms, binEdges = self.approximateDensities(i, X_)
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


    def get_transformed_data(self,ori_data,edge_means,i):
        """
        Transform data points compatible with interpolator boundaries.

        :params: ori_data:
            Original data points

        :params: edge_means:
            Fit original data to boundaries of the histogram bins edges
        """
        ind = self.jj[i]
        Lower, Upper = np.percentile(ori_data[:,ind], [2.5,97.5])
        transformeddata = np.clip(ori_data[:,ind],Lower, Upper)
        return transformeddata

    def transformer(self,rotatedData):
        """
        Interpolate and return approximate Eigenvectors
        Runs on workers

        :param: rotatedData:
            Data to be transformed to fit into the boundaries of the interpolator
        """

        self.interpolators = []
        transformeddata = np.zeros((rotatedData.shape[0],self.k))
        approxValues = np.zeros((rotatedData.shape[0],self.k))

        for i in range(0,self.k):
            self.interpolators.append(ip.interp1d(self.newEdgeMeans[:,i], self.newg[:,i], fill_value = 'extrapolate', kind = 'linear'))
            transformeddata[:,i] = self.get_transformed_data(rotatedData,self.newEdgeMeans,i)
            approxValues[:,i] = self.interpolators[i](transformeddata[:,i])

        approxValues /= np.sum(approxValues**2, axis=0).astype('float32')
        return approxValues

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
        for i in range(0,self.k):
            allpoints[:,i] = self.get_transformed_data(newX, self.newEdgeMeans,i)
            approxVec[:,i] = self.interpolators[i](allpoints[:,i])
        #for p in range(n):
        #    kpoints = allpoints[p]
        #    for d in range(0,self.k):
        #        val = kpoints[d]
        #        approxVec[p,d] = self.interpolators[d](val)
        approxVec /= np.sum(approxVec**2, axis=0).astype('float32')
        return approxVec


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
        S = newsig
        V = np.diag(np.zeros(n))
        labeled = np.where(y != -1)
        print labeled
        V[labeled, labeled] = self.lagrangian
        # Solve for alpha and use it to compute the eigenfunctions, f.
        A = S + np.dot(np.dot(U.T, V), U)
        if np.linalg.det(A) == 0:
            A = A + np.eye(A.shape[1])*0.000001
        b = np.dot(np.dot(U.T, V), y+1)
        self.V = V
        self.yy = y+1
        alpha = np.linalg.solve(A, b)
        return alpha

    def relabel(self,labels, km):
        """
        Label the data points in an ordered way depending on the ascending
        order of the gaussian means.

        :param: labels:
            GMM predicted labels
        """
        means = np.argsort(km.cluster_centers_.flatten())
        for i in range(0, np.size(labels)):
            labels[i] = np.where(means == labels[i])[0][0]
        return labels

    def solver(self ,vectors, alpha):
        """
        f =  U * alpha
        Get and return approximate eigenvectors from eigenfunctions using alpha.

        :param: vectors:
            Approximate eigenfunctions
        """

        U = vectors
        f = np.dot(U, alpha)
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
                return pairwise.rbf_kernel(X, X, gamma = self.gamma)
            else:
                return pairwise.rbf_kernel(X, y, gamma = self.gamma)

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
        n,self.classes = self.getParams(X,y)
        self.labeledX = X[np.where(y !=-1)]
        self.labeledy = y[np.where(y!=-1)]
        rotatedData = self.rotate(X)
        self.newsig, self.newg, self.newEdgeMeans = self.getKSmallest(rotatedData)
        approxValues = self.transformer(rotatedData)
        self.interpolated = approxValues
        self.alpha = self.getAlpha(approxValues, y, n, self.newsig)
        efunctions = self.solver(approxValues,self.alpha)
        self.smooth_functions = efunctions
        #self.gmm = mixture.GMM(n_components = np.size(self.classes),n_iter=5000, covariance_type='diag',min_covar=0.0000001)
        #self.gmm.fit(efunctions)
        #labels = self.gmm.predict(efunctions)
        self.kmeans = KMeans(n_clusters=np.size(self.classes), random_state=0, max_iter = 2000).fit(efunctions)
        labels = self.kmeans.predict(efunctions)
        self.labels_ = self.relabel(labels, self.kmeans)
        return self

    def predict(self,X,y=None):
        """
        Interpolate data and get approximate eigenvectors using alpha.
        Predict labels for the eigenvectors using GMM.

        :params: X:
            Data points whose labels are to be predicted
        :params: y:
            Ground Truth for data points. Label array
        """
        y = np.array([-1] * len(X) + self.labeledy.tolist())
        X = np.array(X.tolist() + self.labeledX.tolist())
        newX = self.pca.transform(X)
        self.n = newX.shape[0]
        approxVec = self.getVectors(newX,self.n)
        testAlpha = self.getAlpha(approxVec, y, self.n, self.newsig)
        newfunctions = self.solver(approxVec, testAlpha)
        self.func = newfunctions
        self.talpha = testAlpha
        #gmm = mixture.GMM(n_components = np.size(self.classes),n_iter=5000, covariance_type='diag',min_covar=0.0000001)
        #newlabels = self.gmm.predict(newfunctions)
        kmeans = KMeans(n_clusters=np.size(self.classes), random_state=0, max_iter = 2000).fit(newfunctions)
        kpredicted =kmeans.predict(newfunctions)
        predictedLabels = self.relabel(kpredicted, kmeans)
        return predictedLabels[:-len(self.labeledy)]
