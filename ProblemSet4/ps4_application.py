import numpy as np
import pylab as pl
import random
# import matplotlib as pl
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.lines import Line2D
import scipy.io
import os
import sys
import matplotlib.pyplot as plt
import scipy.linalg as la
import itertools as it
import time
import pylab as pl


import ps4_implementation as imp


def zero_one_loss(y_true, y_pred):
    n = y_true.shape[0]
    return (1/n)*np.count_nonzero(y_true!=np.sign(y_pred)) # y_true +1,-1 -> y_pred +1,0,-1
    ''' your header here!
    '''

def mean_absolute_error(y_true, y_pred):
    return (1 / y_true.shape[0]) * np.sum(np.abs(y_pred - y_true))

def cv(X, y, method, params, loss_function=mean_absolute_error, nfolds=10, nrepetitions=5):
    zeitanfang = time.time()
    ''' your header here!
    '''
    avErr = 0
    parDim = [len(params[x]) for x in params]
    errList = []
    metList = []
    X_old = X
    y_old = y

    for repetition in range(nrepetitions):
        print("repetition",repetition)
        Idx = np.arange(0,y.shape[0])
        np.random.shuffle(Idx)
        I = np.concatenate(np.array(np.array_split(Idx, nfolds,axis=0)))
        #print("X vor",X.shape,"y vor ",y.shape)
        print("I",I.shape,I)
        print("X",X.shape,X)

        if X.shape[1] == y.shape[0]: X = X.T
        X = X[I]
        y = y[I]
        #print("X",X.shape,"y",y.shape)
        for i in range(I.shape[0]):
            Xtest = X[i]
            ytest = y[i]
            Xtrain = X[I!=I[i]]
            ytrain = y[I!=I[i]]
                    

            for count, j in enumerate(it.product(*params.values())):

                if repetition == 0:
                    met = cross_validate(method, [*j], loss_function, nfolds, nrepetitions, Xtrain ,ytrain, Xtest,ytest)
                    metList.append(met)
                    errList.append(met.cvloss)
                else:
                    met = cross_validate( method, [*j], loss_function, nfolds, nrepetitions, Xtrain ,ytrain, Xtest,ytest)
                    errList[count] += met.cvloss
        X = X_old
        y = y_old
    if len(errList) == 1:
        return errList[0]

    argmin = np.argmin(errList)
    zeitende = time.time()
    print("Dauer Programmausführung:",)
    print(zeitende-zeitanfang)
    return metList[argmin]

def cross_validate(method, paramList, loss_function, nfolds, nrepetitions, Xtrain, ytrain, Xtest, ytest):
    training = method(paramList[0],paramList[1],paramList[2])
    training.fit(Xtrain, ytrain)
    ypred = training.predict(Xtest)
    training.cvloss = loss_function(ytest, ypred)
    return training

def roc_curve(fpr,tpr,start,ende,schritt):
    plt.step(fpr,tpr)
    plt.title("ROC-Curve  Bias : start = "+str(start)+" ende = "+str(ende)+" schritt = "+str(schritt))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def roc_fun(y_true, y_pred,cv_results,X_train,y_train,X_test):
    ''' your header here!
    calculate the TPR and FPR
    
    PRdict = {
    (1,1): 'True Positive',
    (-1,1): 'False Positive',
    (1,-1): 'False Negative',
    (-1,-1): 'True Negative'}
    '''
    TP,FP,FN,TN = 0,0,0,0
    Rates = []
    FPRates,TPRates = [],[]
    M = imp.svm_qp(kernel='gaussian',kernelparameter = cv_results.kernelparameter, C = cv_results.C)
	#M = imp.svm_qp(kernel = gaussian,cv_results.kernelparameter,cv_results.C.)

    biasstart = 0
    biasende = 10
    biasStepSize = 0.10
    biases= np.arange(biasstart,biasende,biasStepSize)
    for bias in biases:
    	M.fit(X_train,y_train)
    	M.b = bias
    	y_pred = M.predict(X_test)
    	loss = float(np.sum(np.sign(y_true) != np.sign(y_pred)))/float(len(y_true))
    	for i  in range(y_pred.shape[0]):

    		#if y_pred[i]+bias >=0:
    		#	y_pred[i] = 1
    		#else:
    		#	y_pred[i] = -1
    		t = (int(y_true[i]),int(y_pred[i]))
    		if t == (1,1):
    			TP = TP +1
    		if t == (-1,1):
    			FP = FP +1
    		if t == (1,-1):
    			FN = FN +1
    		if t == (-1,-1):
    			TN = TN +1
    	TPR = TP/(TP+FN)
    	FPR = FP/(FP+TN)
    	FPRates.append(FPR)
    	TPRates.append(TPR)
    Rates.append((TPR,FPR))
    roc_curve(FPRates,TPRates,biasstart,biasende,biasStepSize)
    return biases[np.argmin(Rates)]







def kmeans(X,k,max_iter=100):
    """ Input: X: (d x n) data matrix with each datapoint in one column
               k: number of clusters
               max_iter: maximum number of iterations
        X 9x2 , k =3 -> mu 3x2 -> r 9 
        Output: mu: (d x k) matrix with each cluster center in one column
                r: assignment vector   """

    #initialization
    n,d= X.shape
    rng = np.random.default_rng()
    mu = rng.choice(X,(k),replace=False)
    r ,r_new = np.zeros(n), np.zeros(n)

    for i in range(max_iter):
        print("iteration = ",i)

        # find nearest cluster center
        for j in range(n):
            r_new[j] = np.argmin(np.linalg.norm(X[j,:]-mu,axis=1)**2)

        # compute nearest cluster center
        for t in range(k):
            print("r_new",r_new)
            if (X[r_new==t].size >0):
                mu[t,:] = np.mean(X[r_new==t],axis=0)
            else:
                # take a random cluster-center
                mu[t,:] = np.mean(rng.choice(X,(k),replace=False))
            #plot every calculation step
            #plt.scatter(X[t==r_new,0],X[t==r_new,1],label=t)
        #plt.scatter(mu[:,0],mu[:,1],c="black",label="mean")
        #plt.legend()
        #plt.show()
        # cluster center didnt change
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


def kmeans_crit(X, r):
    """ Computes k-means criterion

    Input: 
    X: (d x n) data matrix with each datapoint in one column
    r: assignment vector

    Output:
        value: scalar for sum of euclidean distances to cluster centers
    """
    #X=X.T
    n = X.shape[0]
    Loss=0
    for label in np.unique(r):
        delta = np.argwhere(r==label).flatten() 
        # n =
        #tmp = X[delta,i] # X[i][delta] X[i]- >(1,2,3,4,5,6)[0,3,5] -> (1,4,6)
        print("X",X.shape)
        print("delta",delta.shape)
        mu = np.mean(X[delta],axis=0)
        Loss += np.sum(np.linalg.norm( mu)**2,axis=0)
    return Loss



#assignment 4
def assignment_4():
	data = np.load('./data/easy_2d.npz')
	X_test = data['X_te'].T # 100,2
	X_train = data['X_tr'].T #100,2
	y_test = data['Y_te'].T #100
	y_train = data['Y_tr'].T # 100
	
	Sigmas = np.arange(0.1,1.1,0.1)
	C = np.arange(-10,10.1)#np.logspace(-7, 0, num=8, base=10)
	params = {'kernel': ['gaussian'], 'kernelparameter': Sigmas, 'regularization': C}
	
	M = imp.svm_qp # sigmal 0.5 c =5 gar nicht so schlecht
	cv_results = cv(X_train, y_train, M, params, loss_function=mean_absolute_error, nfolds=10, nrepetitions=2)
	#cv_results = cross_validate(M, X_train, y_train,cv=3)
	M = imp.svm_qp(kernel='gaussian',kernelparameter = cv_results.kernelparameter, C = cv_results.C)
	#M = imp.svm_qp(kernel = gaussian,cv_results.kernelparameter,cv_results.C.)
	M.fit(X_train,y_train)
	Y_pred = M.predict(X_test)
	loss = float(np.sum(np.sign(y_test) != np.sign(Y_pred)))/float(len(y_test))
	imp.plot_boundary_2d(X_train, y_train, M,"optimal")
	biases = roc_fun(y_test, Y_pred,cv_results,X_train,y_train,X_test)



	M = imp.svm_qp(kernel='gaussian',kernelparameter = 1.1, C = 0)
	#M = imp.svm_qp(kernel = gaussian,cv_results.kernelparameter,cv_results.C.)
	M.fit(X_train,y_train)
	Y_pred = M.predict(X_test)
	loss = float(np.sum(np.sign(y_test) != np.sign(Y_pred)))/float(len(y_test))
	imp.plot_boundary_2d(X_train, y_train, M,"underfitting")

	M = imp.svm_qp(kernel='gaussian',kernelparameter = 0.2, C = 1)
	#M = imp.svm_qp(kernel = gaussian,cv_results.kernelparameter,cv_results.C.)
	M.fit(X_train,y_train)
	Y_pred = M.predict(X_test)
	loss = float(np.sum(np.sign(y_test) != np.sign(Y_pred)))/float(len(y_test))
	imp.plot_boundary_2d(X_train, y_train, M,"overfitting")

	print("optimal parameters")
	print("cv_results.kernelparameter, C = cv_results.C",cv_results.kernelparameter, cv_results.C)

def assignment_5():
    data = np.load('./data/iris.npz')
    X = data['X'].T # 135,4 -> 3 Klassen 
    Y = data['Y'].T #(135,) -> kmeans 4 klassen ?
    #plt.scatter(X[:,0],X[:,1],X[:,2],X[:,3])
    #plt.show()

    mu,r ,loss = kmeans(X,3,100)
    print("mu,r,loss",mu,mu,r,r.shape,loss)
    
    X0 = X[r==0]
    Y0 = Y[r==0]
    X1 = X[r==1]
    Y1 = Y[r==1]
    X2 = X[r==2]
    Y2 = Y[r==2]

    print("X0,X1,X2",X0.shape,X1.shape,X2.shape)
    X = np.hstack((X0.T,X1.T,X2.T)).T #np.array([X0,X1,X2])
    Y = np.hstack((Y0.T,Y1.T,Y2.T)).T

    Sigmas = np.arange(0.1,1.1,0.01)
    C = np.arange(-10,10,1)#np.logspace(-7, 0, num=8, base=10)
    params = {'kernel': ['linear'], 'kernelparameter': Sigmas, 'regularization': C}
    
    #M = imp.svm_qp # sigmal 0.5 c =5 gar nicht so schlecht
    #cv_results = cv(X, Y, M, params, loss_function=mean_absolute_error, nfolds=10, nrepetitions=2)
    for sigma,C in zip(Sigmas,C):
        M = imp.svm_qp(kernel='linear',kernelparameter = sigma, C = C)
        #M = imp.svm_qp(kernel = gaussian,cv_results.kernelparameter,cv_results.C.)
        M.fit(X,Y)
        #Y_pred = M.predict(X_test)
        #loss = float(np.sum(np.sign(y_test) != np.sign(Y_pred)))/float(len(y_test))
        imp.plot_boundary_2d(X, Y, M,"overfitting")
