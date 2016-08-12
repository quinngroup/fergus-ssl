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
    def __init__(self, kernel = 'rbf', k = -1, gamma = None, lagrangian = 10):
        # This doesn't iterate at all, so the parameters are very different
        # from the BaseLabelPropagation parameters.
        self.kernel = kernel
        self.k = k
        self.gamma = gamma
        #self.n_neighbors = n_neighbors
        self.lagrangian = lagrangian


    def fit(self,X,y=None):
        self.X_ = X
        self.classes = np.sort(np.unique(y))
        dim = self.X_.shape[1]
        if self.classes[0] == -1:
            self.classes = np.delete(self.classes, 0) # remove the -1 from this list
            if self.k == -1:
                self.k = np.size(self.classes)
        randomizedPCA = pca.RandomizedPCA(n_components=dim)
        rotatedData = randomizedPCA.fit_transform(self.X_)
        '''
        sig = an array to save the k smallest eigenvalues that we get for every p(s)
        g   = a 2d column array to save the k smallest eigenfunctions that we get for every p(s)
        '''
        numBins = int(np.sqrt(len(rotatedData)))
        numBins = self.k

        self.sig = np.empty((dim,self.k))
        self.g = np.empty(((dim,numBins,self.k)))
        hist = np.empty((dim,numBins))
        self.b_edgeMeans = np.empty((dim,numBins))
        self.approxValues = np.empty((dim,self.X_.shape[0]))
        transformeddata = np.empty((dim,self.X_.shape[0]))
        self.interpolators=[]
        #sig=np.array([])
        #g=np.array([])
        for i in range(dim):
            histograms,binEdges = np.histogram(rotatedData[:,i],bins=numBins,density=True)
            #add 0.35 to histograms and normalize it
            histograms = histograms+ 0.01
            #histograms /= histograms.sum()
            # calculating means on the bin edges as the x-axis for the interpolators
            print "hist shape: "+str(histograms.shape)
            self.b_edgeMeans[i,:] = np.array([binEdges[j:j + 2].mean() for j in range(binEdges.shape[0] - 1)])
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
            print str(np.real(sigmaVals)[0:self.k])
            self.sig[i,:] = np.real(sigmaVals)[0:self.k]
            self.g[i,:,:] = np.real(functions)[:,0:self.k]
            hist[i,:] = histograms
            #interpolate in 1-D
            self.interpolators.append(ip.interp1d(np.sort(self.b_edgeMeans[i,:]), self.g[i,:, 1]))
            self.interp = ip.interp1d(self.b_edgeMeans[i,:], self.g[i,:, 1])
            '''
            #First check if the original datapoints need to be scaled according to the interpolator ranges
            if(self.X_[:,i].min() < b_edgeMeans[i].min() or self.X_[:,i].max() > b_edgeMeans[i].max()):
                ls=[]
                for num in self.X_[:,i]:
                    ls.append(((((b_edgeMeans[i,:].max()-0.000001)-b_edgeMeans[i,:].min())*(num - self.X_[:,i].min()))/(self.X_[:,i].max() - self.X_[:,i].min())) + b_edgeMeans[i,:].min())
                transformeddata[i,:] = np.array(ls)
                #transformeddata[i,:] = transformer(b_edgeMeans[i].min(), b_edgeMeans[i].max(), self.X_[:,i])
            else:
                print("within range")
                transformeddata[i,:] = np.transpose(self.X_)[i,:]
            '''
            transformeddata[i,:] = self.get_transformed_data(self.X_,self.b_edgeMeans,i)
            self.orig_t = transformeddata
            #get approximated eigenvectors for all n points using the interpolators
            self.approxValues[i,:] = self.interpolators[i](transformeddata[i,:])
        # U: k eigenvectors corresponding to smallest eigenvalues. (n_samples by k)
        # S: Diagonal matrix of k smallest eigenvalues. (k by k)
        # V: Diagonal matrix of LaGrange multipliers for labeled data, 0 for unlabeled. (n_samples by n_samples)
        U = np.transpose(self.approxValues)
        S = np.diag(self.sig[:,1])
        V = np.diag(np.zeros(np.size(y)))
        labeled = np.where(y != -1)
        V[labeled, labeled] = self.lagrangian
        # Solve for alpha and use it to compute the eigenfunctions, f.
        A = S + np.dot(np.dot(U.T, V), U)
        if np.linalg.det(A) == 0:
            A = A + np.eye(A.shape[1])*0.000001
        b = np.dot(np.dot(U.T, V), y)
        self.alpha = np.linalg.solve(A, b)
        print "this is alpha " + str(self.alpha)
        self.labels_ = self.solver(U)
        return self
        #return (sig,g,np.array(interpolators),b_edgeMeans,np.transpose(approxValues))

    def solver(self ,vectors):
        U = vectors
        f = np.dot(U, self.alpha)
        self.func = f
        f = f.reshape((f.shape[0],-1))
        # Set up a GMM to assign the hard labels from the eigenfunctions.
        g = mixture.GMM(n_components = np.size(self.classes))
        g.fit(f)
        #secondEig = self.U_[:,1].reshape((self.U_.shape[0],-1))
        #g.fit(secondEig)
        labels_ = g.predict(f)
        means = np.argsort(g.means_.flatten())
        for i in range(0, np.size(labels_)):
            labels_[i] = np.where(means == labels_[i])[0][0]
        return labels_

    def predict(self,X,y=None):
        dim = X.shape[1]
        self.transformed = np.empty((dim,X.shape[0]))
        approxVectors = np.empty((dim,X.shape[0]))
        for i in range(dim):
            #transform new data
            self.transformed[i,:] = self.get_transformed_data(X,self.b_edgeMeans,i)
            #get approximated eigenvectors for all n points using the interpolators
            approxVectors[i,:] = self.interpolators[i](self.transformed[i,:])
        newlabels = self.solver(np.transpose(approxVectors))
        return newlabels

    def get_transformed_data(self,ori_data,edge_means,i):
        dim = ori_data.shape[1]
        transformeddata = np.empty((dim,ori_data.shape[0]))
        if(ori_data[:,i].min() < edge_means[i].min() or ori_data[:,i].max() > edge_means[i].max()):
            ls=[]
            for num in ori_data[:,i]:
                val = (((edge_means[i,:].max()-edge_means[i,:].min())*(num - ori_data[:,i].min()))/(ori_data[:,i].max() - ori_data[:,i].min())) + edge_means[i,:].min()
                if num==ori_data[:,i].min():
                    val = val + 0.001
                if num==ori_data[:,i].max():
                    val = val - 0.001
                ls.append(val)
            return np.array(ls)
            #transformeddata[i,:] = transformer(b_edgeMeans[i].min(), b_edgeMeans[i].max(), self.X_[:,i])
        else:
            print("within range")
            return np.transpose(ori_data)[i,:]


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
