import numpy as np
import scipy.linalg as LA
import scipy.sparse
import sklearn.utils.arpack as SLA
from sklearn.base import ClassifierMixin
from sklearn.base import BaseEstimator
from sklearn.manifold import spectral_embedding
import sklearn.metrics.pairwise as pairwise
from sklearn import decomposition as pca
from scipy import interpolate as ip
import sklearn.mixture as mixture
import sys
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.neighbors import DistanceMetric
class FergusPropagation(BaseEstimator, ClassifierMixin):
    '''
    The Fergus et al 2009 eigenfunction propagation classifier.

    Parameters
    ----------
    kernel : {'rbf', 'img', 'knn'} #for now just rbf
        String identifier for kernel function to use.
        rbf: Creates a fully-connected dense RBF kernel.
        img: Creates a sparse RBF kernel where each pixel is connected only
            to its 8-neighborhood; all others are 0. ONLY use for image data!
            NOTE: Currently only support .png images.
        knn: Creates RBF kernel with connections to closest n_neighbors points,
            measured in euclidean distance.
    k : integer > 0
        Number of eigenvectors to use (default: # of labels, not including
        unlabeled data) in eigen-decomposition.
    gamma : float > 0
        Parameter for RBF kernel (modes: rbf, img) (default: 20).
    n_neighbors : int > 0
        Parameter for the KNN kernel (modes: knn).
    lagrangian : float > 0
        Lagrangian multiplier to constrain the regularization penalty (default: 10).
    img_dims : array-like, shape = [width, height]
        Tuple specifying the width and height (or COLS and ROWS, respectively)
        of the input image. Used only with a 'img' kernel type.

    Examples
    --------
    TODO

    References
    ----------
    Rob Fergus, Yair Weiss, Antonio Torralba. Semi-Supervised Learning in
    Gigantic Image Collections (2009).
    http://eprints.pascal-network.org/archive/00005636/01/ssl-1.pdf
    '''
    def __init__(self, kernel = None, k = -1,numBins = -1 ,lagrangian = 10):
        # This doesn't iterate at all, so the parameters are very different
        # from the BaseLabelPropagation parameters.
        self.kernel = kernel
        self.k = k
        self.numBins = numBins
        self.lagrangian = lagrangian



    def fit(self,X,y=None):
        self.classes = np.sort(np.unique(y))
        if self.classes[0] == -1:
            self.classes = np.delete(self.classes, 0) # remove the -1 from this list
            if self.k == -1:
                self.k = np.size(self.classes)
        if self.numBins == -1:
            self.numBins = self.k + 1
        self.g = mixture.GMM(n_components = np.size(self.classes),n_iter=5000, covariance_type='diag',min_covar=0.0000001)
        try:
            self.pca_data = pca.PCA(n_components=X.shape[1])
            rotatedData = self.pca_data.fit_transform(X)
        except:
            print("PCA components should not be less than min(samples, n_features) and more that max(samples, n_features)",sys.exc_info()[0])
            raise
        self.X_ = rotatedData
        dim = self.X_.shape[1]
        '''
        newsig = an array to save the k smallest eigenvalues that we get for every p(s)
        newg   = a 2d column array to save the k smallest eigenfunctions that we get for every p(s)
        '''
        #numBins = int(np.sqrt(len(rotatedData)))
        numBins = self.numBins
        sig = np.zeros(dim)
        g = np.zeros((numBins,dim))
        b_edgeMeans = np.zeros((numBins,dim))
        self.interpolators = []
        for i in range(dim):
            histograms,binEdges = np.histogram(self.X_[:,i],bins=numBins,density=True)
            #add 0.01 to histograms and normalize it
            histograms = histograms+ 0.01
            #histograms = histograms / np.linalg.norm(histograms)
            histograms /= histograms.sum()
            # calculating means on the bin edges as the x-axis for the interpolators
            b_edgeMeans[:,i] = np.array([binEdges[j:j + 2].mean() for j in range(binEdges.shape[0] - 1)])
            #get D~, P, W~
            '''
            Wdis = Affinity between discrete points.
            Since Affinity matrix will be build on one histogram at a time. I am using pairwise linear- kernel affinities
            P  = Diagonal matrix of histograms
            Ddis = Diagonal matrix whose diagonal elements are the sum of the columns of PW~P
            Dhat = Diagonal matrix whose diagonal elements are the sum of columns of PW~
            '''
            kernel = "linear"
            Wdis = self._get_kernel(histograms.reshape(histograms.shape[0],1),y=None,ker="linear")
            P  = np.diag(histograms)
            #print("Wdis:" + repr(Wdis.shape) + " P: "+ repr(P.shape))
            Ddis = np.diag(np.sum((P.dot(Wdis.dot(P))),axis=0))
            Dhat = np.diag(np.sum(P.dot(Wdis),axis=0))
            #Creating generalized eigenfunctions and eigenvalues from histograms.
            sigmaVals, functions = scipy.linalg.eig((Ddis-(P.dot(Wdis.dot(P)))),(P.dot(Dhat)))
            arg = np.argsort(np.real(sigmaVals))[1]
            sig[i] = np.real(sigmaVals)[arg]
            g[:,i] = np.real(functions)[:,arg]

        if np.isnan(np.min(sig)):
            nan_num = np.isnan(sig)
            sig[nan_num] = 0
            '''
            First check if the original datapoints need to be scaled according to the interpolator ranges
            '''
        #get approximated eigenvectors for all n points using the interpolators
        self.newsig = np.zeros(self.k)
        self.newg = np.zeros((numBins,self.k))
        self.newEdgeMeans = np.zeros((numBins,self.k))
        self.transformeddata = np.zeros((self.X_.shape[0],self.k))
        self.approxValues = np.zeros((self.X_.shape[0],self.k))
        #selecting the first k eigenvalues and corresponding eigenvectors from all dimensions
        ind =  np.argsort(sig)[0:self.k]
        #print ind
        self.newsig = sig[ind]
        self.newg = g[:,ind]
        self.newEdgeMeans = b_edgeMeans[:,ind]
        for i in range(0,self.k):
            self.interpolators.append(ip.interp1d(self.newEdgeMeans[:,i], self.newg[:,i]))
            self.transformeddata[:,i] = self.get_transformed_data(self.X_,self.newEdgeMeans,i)
            self.approxValues[:,i] = self.interpolators[i](self.transformeddata[:,i])
        # U: k eigenvectors corresponding to smallest eigenvalues. (n_samples by k)
        # S: Diagonal matrix of k smallest eigenvalues. (k by k)
        # V: Diagonal matrix of LaGrange multipliers for labeled data, 0 for unlabeled. (n_samples by n_samples)
        U = self.approxValues
        S = np.diag(self.newsig)
        V = np.diag(np.zeros(np.size(y)))
        labeled = np.where(y != -1)
        V[labeled, labeled] = self.lagrangian
        # Solve for alpha and use it to compute the eigenfunctions, f.
        A = S + np.dot(np.dot(U.T, V), U)
        if np.linalg.det(A) == 0:
            A = A + np.eye(A.shape[1])*0.000001
        b = np.dot(np.dot(U.T, V), y)
        self.alpha = np.linalg.solve(A, b)
        efunctions = self.solver(U)
        # Set up a GMM to assign the hard labels from the eigenfunctions.
        self.g.fit(efunctions)
        self.labels_ = self.g.predict(efunctions)
        self.means = np.argsort(self.g.means_.flatten())
        for i in range(0, np.size(self.labels_)):
            self.labels_[i] = np.where(self.means == self.labels_[i])[0][0]
        return self

    def solver(self ,vectors):
        U = vectors
        f = np.dot(U, self.alpha)
        f = f.reshape((f.shape[0],-1))
        return f

    def predict(self,X,y=None):

        newX = self.pca_data.transform(X)
        approxVec = np.zeros((newX.shape[0],self.k))
        allpoints = np.zeros((newX.shape[0],self.k))
        #Transform all test points into interpolator range
        for i in range(self.k):
            allpoints[:,i] = self.get_transformed_data(newX[:,0:self.k], self.newEdgeMeans,i)

        for p in range(newX.shape[0]):
            kpoints = allpoints[p]
            for d in range(0,self.k):
                val = kpoints[d]
                approxVec[p,d] = self.interpolators[d](val)

        newfunctions = self.solver(approxVec)
        newlabels = self.g.predict(newfunctions)
        for i in range(0, np.size(newlabels)):
            newlabels[i] = np.where(self.means == newlabels[i])[0][0]
        return newlabels

    def get_transformed_data(self,ori_data,edge_means,i):
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
            #transformeddata[i,:] = transformer(b_edgeMeans[i].min(), b_edgeMeans[i].max(), self.X_[:,i])
        else:
            print("within range")
            return ori_data[:,i]


    def _get_kernel(self, X, y = None,ker=None):

        if ker == "rbf":
            if y is None:
                return pairwise.rbf_kernel(X, X, gamma = self.gamma)
            else:
                return pairwise.rbf_kernel(X, y, gamma = self.gamma)
        elif ker == "linear":

            dist = DistanceMetric.get_metric('euclidean')
            if y is None:
                return pairwise.euclidean_distances(X, X)
                #return dist.pairwise(X)
            else:
                return pairwise.euclidean_distances(X, y)
                #return dist.pairwise(X)
        else:
            raise ValueError("%s is not a valid kernel. Only rbf and knn"
                             " are supported at this time" % self.kernel)
