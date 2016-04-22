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

class FergusPropagation(BaseEstimator, ClassifierMixin):
    '''
    The Fergus et al 2009 eigenfunction propagation classifier.

    Parameters
    ----------
    kernel : {'rbf', 'img', 'knn'}
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
    def __init__(self, kernel = 'rbf', k = -1, gamma = 0.1, n_neighbors = 7, lagrangian = 10, img_dims = (-1, -1), numBins = 10):
        # This doesn't iterate at all, so the parameters are very different
        # from the BaseLabelPropagation parameters.
        self.kernel = kernel
        self.k = k
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.lagrangian = lagrangian
        self.img_dims = img_dims
        self.numBins = numBins

    def eigFunc(self,X,y):
        self.X_ = X
        classes = np.sort(np.unique(y))
        if classes[0] == -1:
            classes = np.delete(classes, 0) # remove the -1 from this list
            if self.k == -1:
                self.k = np.size(classes)
        randomizedPCA = pca.RandomizedPCA(n_components=self.k)
        rotatedData = randomizedPCA.fit_transform(self.X_)
        sz = self.X_[:,0].size

        '''
        sig = an array to save the k smallest eigenvalues that we get for every p(s)
        g   = a 2d column array to save the k smallest eigenfunctions that we get for every p(s)
        '''
        sig = np.empty((self.k,self.k))
        g = np.empty(((self.k,self.numBins,self.k)))
        hist = np.empty((self.k,self.numBins))
        b_edgeMeans = np.empty((self.k,self.numBins))
        interpolators = []
        approxValues = np.empty((self.k,self.X_.shape[0]))
        transformeddata = np.empty((self.k,self.X_.shape[0]))
        #sig=np.array([])
        #g=np.array([])
        for i in range(self.k):
            histograms,binEdges = np.histogram(rotatedData[:,i],bins=self.numBins,density=True)
            #add 0.01 to histograms and normalize it
            histograms = histograms+ 0.01
            histograms /= histograms.sum()
            # calculating means on the bin edges as the x-axis for the interpolators
            b_edgeMeans[i,:] = np.array([binEdges[j:j + 2].mean() for j in range(binEdges.shape[0] - 1)])
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
            print("Wdis:" + repr(Wdis.shape) + " P: "+ repr(P.shape))
            Ddis = np.diag(np.sum((P.dot(Wdis.dot(P))),axis=0))
            Dhat = np.diag(np.sum(P.dot(Wdis),axis=0))
            #Creating generalized eigenfunctions and eigenvalues from histograms.
            sigmaVals, functions = scipy.linalg.eig((Ddis-(P.dot(Wdis.dot(P)))),(P.dot(Dhat)))
            #print("eigenValue"+repr(i)+": "+repr(np.real(sigmaVals[0:self.k]))+"Eigenfunctions"+repr(i)+": "+repr(np.real(functions[:,0:self.k])))
            sig[i,:] = np.real(np.sort(sigmaVals)[0:self.k])
            g[i,:,:] = np.real(np.sort(functions, axis=0)[:,0:self.k])
            hist[i,:] = histograms
            #interpolate in 1-D
            interpolators.append(ip.interp1d(np.sort(b_edgeMeans[i,:]), g[i,:, 1]))
            interp = ip.interp1d(b_edgeMeans[i,:], g[i,:, 1])
            #First check if the original datapoints need to be scaled according to the interpolator ranges
            if(X1[:,i].min() < b_edgeMeans[i].min() or X1[:,i].max() > b_edgeMeans[i].max()):
                ls=[]
                for num in X1[:,i]:
                    ls.append(((((b_edgeMeans[i,:].max()-0.1)-b_edgeMeans[i,:].min())*(num - X1[:,i].min()))/(X1[:,i].max() - X1[:,i].min())) + b_edgeMeans[i,:].min())
                transformeddata[i,:] = np.array(ls)
                #transformeddata[i,:] = transformer(b_edgeMeans[i].min(), b_edgeMeans[i].max(), self.X_[:,i])
            else:
                print("within range")
                transformeddata[i,:] = np.transpose(self.X_)[i,:]
            #get approximated eigenvectors for all n points using the interpolators
            approxValues[i,:] = interp(transformeddata[i,:])
        # U: k eigenvectors corresponding to smallest eigenvalues. (n_samples by k)
        # S: Diagonal matrix of k smallest eigenvalues. (k by k)
        # V: Diagonal matrix of LaGrange multipliers for labeled data, 0 for unlabeled. (n_samples by n_samples)
        U = np.transpose(approxValues)

        S = np.diag(sig[:,1])
        V = np.diag(np.zeros(np.size(y)))
        labeled = np.where(y != -1)
        V[labeled, labeled] = self.lagrangian
        # Solve for alpha and use it to compute the eigenfunctions, f.
        A = S + np.dot(np.dot(U.T, V), U)
        b = np.dot(np.dot(U.T, V), y)
        alpha = LA.solve(A, b)
        f = np.dot(U, alpha)
        f = f.reshape((f.shape[0],-1))
        self.func = f
        # Set up a GMM to assign the hard labels from the eigenfunctions.
        g = mixture.GMM(n_components = np.size(classes))
        self.gdash = g
        g.fit(f)
        #secondEig = self.U_[:,1].reshape((self.U_.shape[0],-1))
        #g.fit(secondEig)
        self.labels_ = g.predict(f)
        means = np.argsort(g.means_.flatten())
        for i in range(0, np.size(self.labels_)):
            self.labels_[i] = np.where(means == self.labels_[i])[0][0]

        return self
        #return (sig,g,np.array(interpolators),b_edgeMeans,np.transpose(approxValues))

    def transformer(fmin,fmax,data):
        '''
        Parameters
        ----------
        fmin : min boundary of the interpolator to which the data min is to be transformed to.

        fmax : max boundary of the interpolator to which the data max is to be transformed to.

        data : transform data points into the interpolator's boundary space

        Returns
        -------
        newpoints : scaled datapoints for the given dimension
        '''
        newpoints=[]
        for num in data:
            newpoints.append(((((fmax-1.0)-fmin)*(num - data.min()))/(data.max() - data.min())) + fmin)
        return np.array(newpoints)

    def fit(self, X, y):
        '''
        Fit a semi-supervised eigenfunction label propagation model.

        All input data is provided in matrix X (labeled and unlabeled), and
        corresponding label vector y with dedicated marker value (-1) for
        unlabeled samples.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Feature vectors for data.
        y : array-like, shape = [n_samples]
            Label array, unlabeled points are marked as -1. All unlabeled
            samples will be assigned labels.

        Returns
        -------
        self : An instance of self.
        '''
        self.X_ = X
        # Construct the graph laplacian from the graph matrix.
        W = self._get_kernel(self.X_,y=None, ker=self.kernel)
        D = np.diag(np.sum(W,axis=(1)))
        L = self._unnormalized_graph_laplacian(D, W)
        #L = W
        # Perform the eigen-decomposition.
        vals = None
        vects = None
        classes = np.sort(np.unique(y))
        if classes[0] == -1:
            classes = np.delete(classes, 0) # remove the -1 from this list
            if self.k == -1:
                self.k = np.size(classes)
        #Creating generalized eigenvectors and eigenvalues.
        vals, vects = scipy.linalg.eig(L,D)
        self.u_ = np.real(vals)
        # Construct some matrices.
        # U: k eigenvectors corresponding to smallest eigenvalues. (n_samples by k)
        # S: Diagonal matrix of k smallest eigenvalues. (k by k)
        # V: Diagonal matrix of LaGrange multipliers for labeled data, 0 for unlabeled. (n_samples by n_samples)
        self.vec = np.real(vects)
        self.U_ = np.real(vects[:,:fp.k])
        S = np.diag(self.u_[:fp.k])
        V = np.diag(np.zeros(np.size(y)))
        labeled = np.where(y != -1)
        V[labeled, labeled] = self.lagrangian
        # Solve for alpha and use it to compute the eigenfunctions, f.
        A = S + np.dot(np.dot(self.U_.T, V), self.U_)
        b = np.dot(np.dot(self.U_.T, V), y)
        alpha = LA.solve(A, b)
        self.al = alpha
        f = np.dot(self.U_, alpha)
        f = f.reshape((f.shape[0],-1))
        self.func = f
        # Set up a GMM to assign the hard labels from the eigenfunctions.
        g = mixture.GMM(n_components = np.size(classes))
        self.gdash = g
        g.fit(f)
        #secondEig = self.U_[:,1].reshape((self.U_.shape[0],-1))
        #g.fit(secondEig)
        self.labels_ = g.predict(f)
        means = np.argsort(g.means_.flatten())
        for i in range(0, np.size(self.labels_)):
            self.labels_[i] = np.where(means == self.labels_[i])[0][0]
        # Done!
        return self

    def _get_kernel(self, X, y = None,ker=None):
        print(ker)
        if ker == "rbf":
            if y is None:
                return pairwise.rbf_kernel(X, X, gamma = self.gamma)
            else:
                return pairwise.rbf_kernel(X, y, gamma = self.gamma)
        elif ker == "linear":
            print(ker)
            if y is None:
                return pairwise.euclidean_distances(X, X)
            else:
                return pairwise.euclidean_distances(X, y)
        else:
            raise ValueError("%s is not a valid kernel. Only rbf and knn"
                             " are supported at this time" % self.kernel)

    def _unnormalized_graph_laplacian(self, A, B):
        '''
        Calculates the unnormalized graph laplacian L=D-W

        Parameters
        ----------
        A : array, shape (n, n)
            diagonal matrix.

        B : array, shape (n, n)
            Symmetric affinity matrix.

        Returns
        -------
        L : array, shape (n, n)
            Unnormalized graph laplacian.
        '''
        L = None

        L = A - B

        return L
