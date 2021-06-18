
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
    ''' your header here!
    '''


def mean_absolute_error(y_true, y_pred):
    return (1 / y_true.shape[0]) * np.sum(np.abs(y_pred - y_true))


def cv(X, y, method, params, loss_function=mean_absolute_error, nfolds=10, nrepetitions=5):
    ''' your header here!
    '''
    # initialize the average error
    avErr = 0
    parDim = [len(params[x]) for x in params]
    errList = []
    metList = []

    for repetition in range(nrepetitions):

        # partition the data
        XandY = np.array([*X.T, y]).T
        np.random.shuffle(XandY)
        partitions = np.array(np.array_split(XandY, nfolds, axis=0))

        for fold in range(nfolds):

            # Create TestSet and TrainingSet
            testSet = partitions[fold]

            # Create the training set
            a = np.arange(nfolds)
            a = a[a != fold]
            trainingSet = np.array(partitions[a])
            trainingSet = trainingSet.reshape(trainingSet.shape[0] * trainingSet.shape[1], trainingSet.shape[2])

            #Shufflen und die Repititions hier erstellen und an cross_validate übergeben...
            for count, i in enumerate(it.product(*params.values())):

                if repetition == 0:
                    met = cross_validate(X, y, method, [*i], loss_function, nfolds, nrepetitions, trainingSet, testSet)
                    metList.append(met)
                    errList.append(met.cvloss)
                else:
                    met = cross_validate(X, y, method, [*i], loss_function, nfolds, nrepetitions, trainingSet, testSet)
                    errList[count] += met.cvloss

    if len(errList) == 1:
        return errList[0]

    argmin = np.argmin(errList)
    return metList[argmin]

#Shufflen und repetion in CV, für alle Settings die gleichen Folds


def cross_validate(X, y, method, paramList, loss_function, nfolds, nrepetitions, trainingSet, testSet):

    # Train and Predict the Data
    Training = method(paramList[0],paramList[1],paramList[2])

    Training.fit(trainingSet[:, :-1], trainingSet[:, -1])
    y_pred = Training.predict(testSet)

    # Compare the true and predicted labels and calculate the error
    y_true = testSet[:,-1]  # [:-2]


    #Use the specified loss function from the parameters
    tempErr = loss_function(y_true, y_pred)
    #tempErr = np.count_nonzero(y_true != y_pred) / y_true.size

    Training.cvloss = tempErr

    return Training



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

    def fit(self, X, y, kernel=False, kernelparameter=False, regularization=False):

        self.X = X

        if kernel is not False:
            self.kernel = kernel
        if kernelparameter is not False:
            self.kernelparameter = kernelparameter
        if regularization is not False:
            self.regularization = regularization

        # Calculate the Kernel Matrix K
        #Fehlerquelle
        self.K = self.calcXYker(X, y,kernel, kernelparameter)

        # Calculate the Regularization term C
        if regularization == 0:

            # Calculate the mean of the Eigenvalues as the center of candidates
            eigVals, eigVecs = np.linalg.eig(self.K)
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
        self.alpha = np.linalg.inv(self.K + self.regularization * np.eye(self.K.shape[0])) @ y

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
        KerMat = self.calcXYker(self.X,Y ,self.kernel, self.kernelparameter)
        return KerMat.T@self.alpha[:,np.newaxis]

    def ker(self, x, y, ker, kernelParameter):
        """
        if  isinstance(ker, list):
            kern = ker[0]
            sigma = ker[1]
            d = ker[2]
        else:
            kern = ker
            sigma = kernelParameter  # ker[1]
            d = kernelParameter  # ker[2]"""
        #print("x",x,"y",y,"ker",ker,"kernelParameter",kernelParameter)

        # Linear Kernel
        if ker == 'linear':
            return x * y

        # Polynomial Kernel
        if ker == 'polynomial':
            d = kernelParameter
            return ((x * y) + 1) ** d

        # Gaussian Kernel
        if ker == 'gaussian':
            sigma = kernelParameter
            exponent = - (np.abs((x - y)) ** 2) / (2 * (sigma ** 2))
            return np.exp(exponent)

    def calcKer(self, X, kernel, kernelparameter):

        # Linear Kernel
        if kernel == 'linear':
            return X @ X.T

        # Polynomial Kernel
        if kernel == 'polynomial':

            d = kernelparameter
            one = np.ones(X.shape)
            return np.linalg.matrix_power((X @ X.T + one), d)

        # Gaussian Kernel
        if kernel == 'gaussian':

            sigma = kernelparameter
            G = X @ X.T
            g = np.diag(G)
            one = np.ones((g.shape[0]))
            distances = np.outer(g, one) + np.outer(one, g) - 2 * G
            params = (-1 / (sigma ** 2)) * distances
            return np.exp(params)

        # "Catch" the wrong specifications
        return X @ X.T


    def calcXYker(self, X,Y, kernel, kernelparameter):
        # Linear Kernel
        if kernel == 'linear':
            K = np.zeros((X.shape[0],Y.shape[0]))
            for i,x in enumerate(X):
                for j,y in enumerate(Y):
                    try:
                        K[i,j] = self.ker( x, y, kernel , 0)#np.exp(-1*np.linalg.norm(x-y)**2)
                    except:
                        K[i,j] = self.ker( x, y, kernel , 0)[0]
                    #K[i,j] = self.ker(x, y, "linear", 0)
            return K

        # Polynomial Kernel
        if kernel == 'polynomial':
            d = kernelparameter
            K = np.zeros((X.shape[0],Y.shape[0]))
            for i,x in enumerate(X):
                for j,y in enumerate(Y):
                    one = np.ones((x@y.T).shape)
                    try:
                        K[i,j] = self.ker( x, y, kernel , d)#np.exp(-1*np.linalg.norm(x-y)**2)
                    except:
                        K[i,j] = self.ker( x, y, kernel , d)[0]
                    #K[i,j] = self.ker( x, y, kernel , d)#(x@y.T+one)**d
            return K
            
            #K = X @ Y.T
            #one = np.ones(K.shape)
            #return (K+ one)**d"""
            """d = kernelparameter
            one = np.ones(X.shape)
            return np.linalg.matrix_power((X @ Y.T + one), d)"""

        # Gaussian Kernel
        if kernel == 'gaussian':
            K = np.zeros((X.shape[0],Y.shape[0]))
            d = kernelparameter
            for i,x in enumerate(X):
                for j,y in enumerate(Y):
                    try:
                        K[i,j] = self.ker( x, y, kernel , d)#np.exp(-1*np.linalg.norm(x-y)**2)
                    except:
                        K[i,j] = self.ker( x, y, kernel , d)[0]
                    #K[i,j] = self.ker( x, y, kernel , d)#np.exp(-1*np.linalg.norm(x-y)**2)
            return K

        # "Catch" the wrong specifications
        return X @ X.T
