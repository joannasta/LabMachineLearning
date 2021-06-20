

""" ps3_implementation.py
PUT YOUR NAME HERE:
Joanna Stamer
Friedrich Wicke
Write the functions
- cv
- zero_one_loss
- krr
Write your implementations in the given functions stubs!
(c) Daniel Bartz, TU Berlin, 2013
"""

import numpy as np
import scipy.linalg as la
import itertools as it
import time
import pylab as pl


# from mpl_toolkits.mplot3d import

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
        Idx = np.arange(0,X.shape[1])
        np.random.shuffle(Idx)
        I = np.array(np.array_split(Idx, nfolds,axis=0))
        #print("X vor",X.shape,"y vor ",y.shape)

        X = X[:,I].T     #shape = (400,2) ->2,400  nicht für (8,468)
        y = y[I].T      #shape = 400
        #print("X",X.shape,"y",y.shape)
        for i in range(I.shape[0]):
            Xtest = X[i]
            ytest = y[i]
            Xtrain = X[I.T!=I.T[i]]
            ytrain = y[I.T!=I.T[i]]
            #print("Xtrain",Xtrain.shape)
            try:
                Xtest = Xtest.reshape(Xtest.shape[0]*Xtest.shape[1],Xtest.shape[2])
                Xtrain = Xtrain.reshape(Xtrain.shape[0]*Xtrain.shape[1],Xtrain.shape[2])
                ytest = ytest.reshape(ytest.shape[0]*ytest.shape[1],)
                ytrain = ytrain.reshape(ytrain.shape[0]*ytrain.shape[1],)
            except:
                Xtest = Xtest.reshape(Xtest.shape[0],Xtest.shape[1])
                Xtrain = Xtrain.reshape(Xtrain.shape[0],Xtrain.shape[1])
                ytest = ytest.reshape(ytest.shape[0],)
                ytrain = ytrain.reshape(ytrain.shape[0],)
                    

            for count, i in enumerate(it.product(*params.values())):

                if repetition == 0:
                    met = cross_validate(method, [*i], loss_function, nfolds, nrepetitions, Xtrain ,ytrain, Xtest,ytest)
                    metList.append(met)
                    errList.append(met.cvloss)
                else:
                    met = cross_validate( method, [*i], loss_function, nfolds, nrepetitions, Xtrain ,ytrain, Xtest,ytest)
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
  





class krr():
    ''' your header here!
    '''

    def __init__(self, kernel='linear', kernelparameter=1, regularization=0):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization
        self.alpha = 0
        self.K = 0
        self.X = 0

    def fit(self, X, y):

        self.X = X

        # Calculate the Kernel Matrix K
        #Fehlerquelle
        self.K = self.calcKer(X)

        # Calculate the Regularization term C
        if self.regularization == 0:

            # Calculate the mean of the Eigenvalues as the center of candidates
            eigVals, eigVecs = np.linalg.eigh(self.K)
            eigMean = np.mean(eigVals)

            # Create a List of candidates
            paramList = np.logspace(-2, 2, num=30, base=eigMean)

            # iterate over the list, find the lowest error and save the C value

            U = eigVecs
            UT = np.linalg.inv(U)
            L = eigVals
            UTY = UT @ y
            errors = [self.regError(C, U, UT, L, y, UTY) for C in paramList]

            # set the reg param of the object
            self.regularization = paramList[np.argmin(errors)]

        # set the weight vector of the object
        self.alpha = np.linalg.solve(self.K + self.regularization * np.eye(self.K.shape[0]),y)
        #self.alpha = np.linalg.inv(self.K + self.regularization * np.eye(self.K.shape[0])) @ y

        return self

    def regError(self, C, U, UT, L, y, UTY):

        # Calculate the "constants"
        n = y.shape[0]
        Linv = np.diag([1 / (C + x) for x in L])
        ULLinv = U @ np.diag(L) @ Linv
        S = ULLinv @ UT
        Sy = ULLinv @ UTY

        # Calculate the error
        eps = 0
        for i in range(n):
            eps += ((y[i] - Sy[i]) / (1 - S[i, i])) ** 2
        eps = eps / n

        return eps


    def predict(self,Y):

        #Wir brauchen neue calcKer Funktion für Kernel aus X und Y mit X!=Y
        KerMat = self.calcKer(self.X ,Y)
        return KerMat.T@self.alpha

    def ker(self, x, y):

        # Linear Kernel
        if self.kernel == "linear":
            return x*y

        # Polynomial Kernel
        if self.kernel == "polynomial":
            d = self.kernelparameter
            return ((x*y) + 1) ** d

        # Gaussian Kernel
        if self.kernel == "gaussian":
            sigma = self.kernelparameter
            exponent = - (np.abs(x)**2 + np.abs(y)**2-2*(x*y)) / (2 * (sigma ** 2))
            return np.exp(exponent)
        
    def calcKer(self,X,Y=None):
        #print("kernel",self.kernel)
        #print("X",X.shape,"kP",self.kernelparameter)
        if Y is None:
            Y = X.T
            D = np.zeros((X.shape[0],Y.shape[1]))
            if self.kernel == "linear":
                for i in range(X.shape[0]):
                    for j in range(Y.shape[1]):
                        D[i,j]= self.ker(X[i,0],Y[0,j])
                    
            if self.kernel == "polynomial":
                for i in range(X.shape[0]):
                    for j in range(Y.shape[1]):
                        D[i,j]= self.ker(X[i,0],Y[0,j])
            if self.kernel == "gaussian":
                for i in range(X.shape[0]):
                    for j in range(Y.shape[1]):
                        D[i,j]= self.ker(X[i,0],Y[0,j])
        else:
            D = np.zeros((X.shape[0],Y.shape[0]))
            if self.kernel == "linear":
                for i in range(X.shape[0]):
                    for j in range(Y.shape[0]):
                        D[i,j]= self.ker(X[i,0],Y[j,0])
                    
            if self.kernel == "polynomial":
                for i in range(X.shape[0]):
                    for j in range(Y.shape[0]):
                        D[i,j]= self.ker(X[i,0],Y[j,0])
            if self.kernel == "gaussian":
                for i in range(X.shape[0]):
                    for j in range(Y.shape[0]):
                        D[i,j]= self.ker(X[i,0],Y[j,0])
            
        return D
  



