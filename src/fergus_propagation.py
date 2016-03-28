import numpy as np
import scipy.linalg as LA
import scipy.sparse
import sklearn.utils.arpack as SLA
from sklearn.base import ClassifierMixin
from sklearn.base import BaseEstimator
from sklearn.manifold import spectral_embedding
import sklearn.metrics.pairwise as pairwise
import sklearn.mixture as mixture

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
    def __init__(self, kernel = 'rbf', k = -1, gamma = 0.005, n_neighbors = 7, lagrangian = 10, img_dims = (-1, -1)):
        # This doesn't iterate at all, so the parameters are very different
        # from the BaseLabelPropagation parameters.
        self.kernel = kernel
        self.k = k
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.lagrangian = lagrangian
        self.img_dims = img_dims

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
        W = self._get_kernel(self.X_)
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

    def _get_kernel(self, X, y = None):
        if self.kernel == "rbf":
            if y is None:
                return pairwise.rbf_kernel(X, X, gamma = self.gamma)
            else:
                return pairwise.rbf_kernel(X, y, gamma = self.gamma)
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
