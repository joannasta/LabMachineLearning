""" ps2_implementation.py

PUT YOUR NAME HERE:
<FIRST_NAME><LAST_NAME>


Write the functions
- kmeans
- kmeans_agglo
- agglo_dendro
- norm_pdf
- em_gmm
- plot_gmm_solution

(c) Felix Brockherde, TU Berlin, 2013
    Translated to Python from Paul Buenau's Matlab scripts
"""
from __future__ import division  # always use float division
import numpy as np
from scipy.spatial.distance import cdist  # fast distance matrices
from scipy.cluster.hierarchy import dendrogram  # you can use this
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # for when you create your own dendrogram
import scipy.io # von joanna

def kmeans(X,k,max_iter=100):
    """ Input: X: (d x n) data matrix with each datapoint in one column
               k: number of clusters
               max_iter: maximum number of iterations
        Output: mu: (d x k) matrix with each cluster center in one column
                r: assignment vector   """
    n,d= X.shape
    mu = np.random.rand(k,d)
    r ,rnew = np.zeros(n), np.zeros(n)
    for i in range(max_iter):
        print("i = ",i)
        for j in range(n):
            rnew[j] = np.argmin(np.linalg.norm(X[j,:]-mu,axis=-1)**2,axis=-1)
        for t in range(k):
            mu[t,:] = np.mean(X[rnew==t],axis=0)
        
        plt.scatter(X[:,0],X[:,1],c="b")
        plt.scatter(mu[:,0],mu[:,1],c="r")
        plt.show()
        if np.all(r == rnew):
            print("number of cluster memberships which changed in the preceding step = ",0)
            print("loss = ",0)
            break
        else:
            print("number of cluster memberships which changed in the preceding step = ",np.size(r==rnew)-np.count_nonzero(r==rnew))
            r = rnew
            loss = kmeans_agglo(X,r)
            print("loss = ",loss)
    return mu, r, loss

def kmeans_agglo(X, r):
    """ Performs agglomerative clustering with k-means criterion

    Input:
    X: (d x n) data matrix with each datapoint in one column
    r: assignment vector

    Output:
    R: (k-1) x n matrix that contains cluster memberships before each step
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """


    def kmeans_crit(X, r):
        """ Computes k-means criterion

        Input: 
        X: (d x n) data matrix with each datapoint in one column
        r: assignment vector

        Output:
        value: scalar for sum of euclidean distances to cluster centers
        """
        X=X.T
        n = X.shape[0]
        Loss=0
        for i in range(n):
                for label in np.unique(r):
                    delta = np.argwhere(r==label).flatten()
                    tmp = X[i,delta]
                    mu = np.mean(tmp,axis=0)
                    Loss += np.sum(np.linalg.norm(tmp - mu)**2,axis=0)
        return Loss
    return kmeans_crit(X,r)
#X = np.array([[1,0,2,3,1,2,4,2,4,1],[1,0,2,3,1,2,4,2,4,1]])
#r = np.array([[1,0,2,3,1,2,4,2,4,1]])

def test_crit():
    # m??glicher fehler falls r mit 2 eckigen Klammern initialisiert wird
    X = np.array([[1,0,2,3,1,2,4,2,4,1],[1,0,2,3,1,2,4,2,4,1]])
    r = np.array([1,0,2,3,1,2,4,2,4,1])
    return(kmeans_agglo(X,r))
def test_kmeans():
    # m??glicher fehler falls r mit 2 eckigen Klammern initialisiert wird
    #X = np.array([[0,0,1,1,2,2,3,3,4,4,5,5],[0,0,1,1,2,2,3,3,4,4,5,5]])
    #r = np.array([1,0,2,3,1,2,4,2,4,1])
    #return(kmeans(X,5))
    X = np.array([[0., 1., 1., 10., 10.25, 11., 10., 10.25, 11.],
                  [0., 0., 1.,  0.,   0.5,  0.,  5.,   5.5,  5.]]).T
    perfect_r = [1,0,1,2,2,1,2,2,2]

    worked1 = False
    worked2 = False

    for _ in range(10):
        mu, r,loss = kmeans(X, k=3)
        if (r[0]==r[1]==r[2]!=r[3] and r[3]==r[4]==r[5]!=r[6] and r[6]==r[7]==r[8]):
            worked1 = True

        # test one cluster center
        print(mu,mu.shape)
        if (np.linalg.norm(mu[0] - [10.41666, 0.1666]) < 0.1 or
            np.linalg.norm(mu[1] - [10.41666, 0.1666]) < 0.1 or
            np.linalg.norm(mu[2] - [10.41666, 0.1666]) < 0.1):
            worked2 = True
        if worked1 and worked2:
            break
    if not worked1:
        raise AssertionError('test_kmeans cluster assignments are wrong.')
    if not worked2:
        raise AssertionError('test_kmeans did not find the correct cluster center.')



def agglo_dendro(kmloss, mergeidx):
    """ Plots dendrogram for agglomerative clustering

    Input:
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """
    mat = scipy.io.loadmat('usps.mat')
    #1) Load the USPS dataset
    L=mat.get("data_labels")
    P=mat.get("data_patterns")
    kmeans(P,10)
    pass


def norm_pdf(X, mu, C):
    """ Computes probability density function for multivariate gaussian

    Input:
    X: (d x n) data matrix with each datapoint in one column
    mu: vector for center
    C: covariance matrix

    Output:
    pdf value for each data point
    """

    pass


def em_gmm(X, k, max_iter=100, init_kmeans=False, eps=1e-3):
    """ Implements EM for Gaussian Mixture Models

    Input:
    X: (d x n) data matrix with each datapoint in one column
    k: number of clusters
    max_iter: maximum number of iterations
    init_kmeans: whether kmeans should be used for initialisation
    eps: when log likelihood difference is smaller than eps, terminate loop

    Output:
    pi: 1 x k matrix of priors
    mu: (d x k) matrix with each cluster center in one column
    sigma: list of d x d covariance matrices
    """

    pass

def plot_gmm_solution(X, mu, sigma):
    """ Plots covariance ellipses for GMM

    Input:
    X: (d x n) data matrix with each datapoint in one column
    mu: (d x k) matrix with each cluster center in one column
    sigma: list of d x d covariance matrices
    """

    pass
