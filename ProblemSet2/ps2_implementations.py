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
import scipy as sp # Frido
import copy # Frido

def kmeans(X,k,max_iter=100):
    """ Input: X: (d x n) data matrix with each datapoint in one column
               k: number of clusters
               max_iter: maximum number of iterations
        X 9x2 , k =3 -> mu 3x2 -> r 9
        Output: mu: (d x k) matrix with each cluster center in one column
                r: assignment vector   """
    n,d= X.shape
    #mu = np.random.rand(k,d)+ np.mean(X,axis=0)
    #mu = X[:k].copy()
    rng = np.random.default_rng()
    mu = rng.choice(X,(k),replace=False)
    r ,r_new = np.zeros(n), np.zeros(n)
    for i in range(max_iter):
        print("iteration = ",i)
        for j in range(n):
            r_new[j] = np.argmin(np.linalg.norm(X[j,:]-mu,axis=1)**2)
        for t in range(k):
            print("r_new",r_new)
            if (X[r_new==t].size >0):
                mu[t,:] = np.mean(X[r_new==t],axis=0)#hier noch mit np.where arbeiten [t,:]
            else:
                mu[t,:] = np.mean(rng.choice(X,(k),replace=False))
            plt.scatter(X[t==r_new,0],X[t==r_new,1],label=t)
        plt.scatter(mu[:,0],mu[:,1],c="black",label="mean")
        plt.legend()
        plt.show()
        #print("r",r,"r_new",r_new)
        #print("mu")
        if np.all(r == r_new):
            print("number of cluster memberships which changed in the preceding step = ",0)
            print("loss = ",0)
            break
        else:
            print("number of cluster memberships which changed in the preceding step = ",np.size(r==r_new)-np.count_nonzero(r==r_new))
            r = r_new.copy()
            loss = kmeans_agglo(X,r)
            print("loss = ",loss)
    return mu, r,loss

def kmeans_agglo(X, r, showSizes = False):
    """ Performs agglomerative clustering with k-means criterion
    Input:
    X: (d x n) data matrix with each datapoint in one column
    r: assignment vector
    Output:
    R: (k-1) x n matrix that contains cluster memberships before each step
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """
    uniques = np.unique(r)
    max     = uniques[-1]
    k       = uniques.shape[0]
    n       = X.shape[0]

    #Handle the case for k<3
    if k<2:
        return

    #initialize the return Matrices
    R           = np.zeros((k-1,n))
    kmloss      = np.zeros(k)
    sizes       = np.zeros(k-1)
    mergeidx    = np.zeros((k-1,2))

    #Save the first Loss value
    kmloss[0] = kmeans_crit(X,r)

    #Compute the clustering steps
    for i in range (k-1):
        #agglomerative step

        #Find the current clusters
        uniques = np.unique(r)

        #Find the two clusters, whose merger causes the lowest cost
        minCost = np.infty
        c1 = c2 = -1
        for count, val in enumerate(uniques):
            for j in range(count+1, uniques.shape[0], 1):
                tempR = [val if x==uniques[j] else x for x in r]
                if kmeans_crit(X,tempR) < minCost:
                    c1, c2 = count, j
                    minCost = kmeans_crit(X,tempR)

        #Update Parameters
        mergeidx[i] = [uniques[c1], uniques[c2]]
        R[i]        = r
        r           = [max+1 if x == uniques[c1] or x == uniques[c2] else x for x in r]
        #R[i]        = r
        kmloss[i+1] = kmeans_crit(X,r)
        max        += 1
        sizes[i]    = np.count_nonzero(r == max)

    if showSizes:
        return R, kmloss, mergeidx, sizes
    return R, kmloss, mergeidx

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



def agglo_dendro(kmloss, mergeidx):
    """ Plots dendrogram for agglomerative clustering
    Input:
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """

    n               = mergeidx.shape[0]
    linkMat         = np.zeros((n,4))
    linkMat[:,0]    = mergeidx[:,0]
    linkMat[:,1]    = mergeidx[:,1]

    #The following doesn't make sense, as we do not have access to the original data.
    #The Sizes should be passed in as a parameter, not calculated here
    #Get the highest original Index
    max = (np.unique(mergeidx.flatten())[-n])

    sizes = np.ones(n)

    for count, val in enumerate(mergeidx):
        v0, v1 = val[0], val[1]
        if v0<max+1 and v1<max+1:
            sizes[count] = 2
        elif v0<max+1:
            sizes[count] = 1 + sizes[int(v1-max-1)]
        elif v1<max+1:
            sizes[count] = 1 + sizes[int(v0-max-1)]
        else:
            sizes[count] = sizes[int(v0-max-1)] + sizes[int(v1-max-1)]

    linkMat[:,3]    = sizes

    for i in range(n):
        linkMat[i,2] = kmloss[i+1]#-kmloss[i]
    print(linkMat)
    dendrogram(linkMat)
    plt.show()
    return linkMat

def norm_pdf(xi, mu, C):
    xi = xi[:,np.newaxis]
    n,d = xi.shape
    xi = xi.T
    try:
        const = 1/((2*np.pi)**(d/2) * np.linalg.det(C)**(1/2))
        xm = xi-mu
        #print("xm",xm.shape,xm)
        term = np.exp((-1/2)* xm@np.linalg.solve(C,xm.T))
        yi = const*term
    except :
        C = C + np.random.normal(0 ,0.1,d)
        const = 1/((2*np.pi)**(d/2) * np.linalg.det(C)**(1/2))
        xm = xi-mu
        #print("xm",xm.shape,xm)
        term = np.exp((-1/2)* xm@np.linalg.solve(C,xm.T))
        yi = const*term
    return yi

def plot_gmm_solution(X, mu, sigmas):
    k = mu.shape[0]
    print("k",k)
    """ Plots covariance ellipses for GMM
    Input:
    X: (d x n) data matrix with each datapoint in one column
    mu: (d x k) matrix with each cluster center in one column
    sigma: list of d x d covariance matrices
    """
    #print("X",X.shape)
    #print("Xsorted",np.sort(X,axis=1))
    #X = np.sort(X,axis=1)
    #plt.scatter(X[:,0],X[:,1])
    #plt.scatter(mu[:,0],mu[:,1])

    #clusters = np.array_split(X, k)
    r_new = np.zeros(X.shape[0])
    for j in range(X.shape[0]):
            r_new[j] = np.argmin(np.linalg.norm(X[j,:]-mu,axis=1)**2)
    print("r_new",r_new)
    clusters = X
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    for i in range(k):
        lambdas,vectors = np.linalg.eig(sigmas[i])
        eigenvalues, eigenvectors = lambdas, vectors
        theta = np.linspace(0, 2*np.pi, 1000);
        ellipsis = (((np.sqrt(eigenvalues[None,:]) * eigenvectors)) @ [np.sin(theta), np.cos(theta)]) + np.repeat([mu[i]], 1000, axis=0).T
        plt.plot(ellipsis[0,:], ellipsis[1,:], color='black')
        ax.scatter(clusters[i==r_new,0],clusters[i==r_new,1],label="cluster"+str(i))#plt.scatter(X[t==r_new,0],X[t==r_new,1],label=t)
    ax.plot(mu[:,0],mu[:,1],'r+',label="em_gmm-mean")
    plt.legend()
    plt.show()

    pass


def em_gmm(X, K, max_iter=30, init_kmeans=True, eps=1e-3):
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
    d = 2
    n = 9
    """
    n,d = X.shape
    pi = np.array([1/K]*K)
    if init_kmeans:
        mu,r,loss = kmeans(X,K)
    else:
        rng = np.random.default_rng()
        mu = rng.choice(X,(K),replace=False)
    sigma = [np.eye(d)]*K
    gamma = np.zeros((K,n))
    gamma_ = np.zeros((K,n))
    eta = np.zeros(K)

    for _ in range(max_iter):
        print("durchgang",_)
        gamma_ = gamma
        for k in range(K):
            for i in range(n):
                dividend = pi[k]* norm_pdf(X[i,:],mu[k],sigma[k])
                divisor = sum([(pi[k_] * norm_pdf(X[i,:], mu[k_], sigma[k_])) for k_ in range(K)])
                gamma[k,i] = dividend/divisor

        for k in range(K):
            eta[k] = sum([gamma[k,i] for i in range(n)])
            pi[k] = eta[k]/n
            mu[k] = 1/eta[k] * sum([(gamma[k,i] * X[i]) for i in range(n)])
            s=0
            for i in range(n):
                tmp = (X[i] - mu[k])[:,np.newaxis]
                s += gamma[k,i] * tmp * tmp.T
            sigma[k] = 1/eta[k] * s
        if np.all(np.abs(np.array(gamma_)-np.array(gamma))<eps):
                break

    loglik = gamma
    plot_gmm_solution(X, mu, sigma)
    return pi,mu,sigma,loglik


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
        print("mu",mu,"r",r)
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
def test_Agglo():
    X = np.array([[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]])
    r = X[0]
    #r[0]=1
    #r[4]=1
    X = X.T
    #print("r",r)
    R, kmloss, mergeidx = kmeans_agglo(X,r)
    print("R",R,"kmloss", kmloss, "mergeidx", mergeidx)
    print(sp.cluster.hierarchy.linkage(X, method = 'centroid'))
    #print(transformMergeIdx(mergeidx, 10))
    agglo_dendro(kmloss, mergeidx)
    #dendrogram(sp.cluster.hierarchy.linkage(X, method = 'centroid'))
    #plt.show()

def testagglotest():#self):
    X = np.array([[0., 1., 1., 10., 10.25, 11., 10., 10.25, 11.],
                  [0., 0., 1.,  0.,   0.5,  0.,  5.,   5.5,  5.]]).T
    #mu, r,  = kmeans(X, k=3)
    #r       = copy.deepcopy(r.flatten())
    #print(X,r,"Xundr")
    r = np.array([0, 2, 2, 1, 1, 1, 1, 1, 1])
    R, kmloss, mergeidx = kmeansagglo(X, r)
    print(r_,"R, kmloss, mergeidx",R, kmloss, mergeidx)

def kmeans_usps_test ():
    mat = scipy.io.loadmat('usps.mat')
    #1) Load the USPS dataset
    L=mat.get("data_labels")
    P=mat.get("data_patterns")
    kmeans(P,10)
    pass
def test_em_gmm():
    X = np.array([[0., 1., 1., 10., 10.25, 11., 10., 10.25, 11.],
                  [0., 0., 1.,  0.,   0.5,  0.,  5.,   5.5,  5.]]).T
    perfect_r = [1,0,1,2,2,1,2,2,2]
    worked1 = False
    worked2 = False
    for _ in range(10):
        mpi, mu, sigma, _ = imp.em_gmm(X, k=3)
        # test one cluster center
        if (np.linalg.norm(mu[0] - [10.41666, 0.1666]) < 0.1 or
            np.linalg.norm(mu[1] - [10.41666, 0.1666]) < 0.1 or
            np.linalg.norm(mu[2] - [10.41666, 0.1666]) < 0.1):
            worked1 = True
        if ((np.abs(np.linalg.det(sigma[0]) - 0.03703) < 0.001 or
            np.abs(np.linalg.det(sigma[1]) - 0.03703) < 0.001 or
            np.abs(np.linalg.det(sigma[2]) - 0.03703) < 0.001) and
            (np.abs(np.linalg.det(sigma[0]) - 0.00925) < 0.001 or
            np.abs(np.linalg.det(sigma[1]) - 0.00925) < 0.0001 or
            np.abs(np.linalg.det(sigma[2]) - 0.00925) < 0.0001)):
            worked2 = True
        if worked1 and worked2:
            imp.plot_gmm_solution(X, mu, sigma)
            break

    if not worked1:
        raise AssertionError('test_em_gmm did not find the correct cluster center.')
    if not worked2:
        raise AssertionError('test_em_gmm did not find the correct cluster center.')
