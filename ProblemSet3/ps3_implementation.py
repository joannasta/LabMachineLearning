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
from mpl_toolkits.mplot3d impor


def zero_one_loss(y_true, y_pred):
    ''' your header here!
    '''


def mean_absolute_error(y_true, y_pred):
    return (1/y_true.shape[0])*np.sum(np.abs(y_pred-y_true))


def cv(X, y, method, params, loss_function=zero_one_loss, nfolds=10, nrepetitions=5):
    ''' your header here!
    '''
    #initialize the average error
    avErr = 0
    parDim = [len(params[x]) for x in params]
    errList = []

    for i in it.product(*params.values()):
        errList.append(cross_validate(X, y, [*i], loss_function, nfolds, nrepetitions , False))
    errMat = np.array(errList).reshape((parDim))

    if errMat.size == 1:
        return errMat[0]

    argMins = np.argmin(errMat)
    optParams = [params[x][argMins[j]] for j, x in enumerate(params)]

    return cross_validate(X, y, optParams, loss_function, nfolds, nrepetitions , True)



def cross_validate(X, y, paramList, loss_function, nfolds, nrepititions, retMet):

    for repetition in range(nrepetitions):

        'Calculate the size of the partitions
        DL = np.concatenate((X,[y]), axis=0).T
        shuffledDL = np.random.shuffle(DL)
        partitions = np.array_split(shuffledDL, nfolds, axis=0)

        for fold in range(nfolds):

            #Create TestSet and TrainingSet
            testSet = partitions[fold]
            trainingSet = np.vstack(np.delete(partitions, fold))

            #Train and Predict the Data
            Training = method(paramList)
            Training.fit(trainingSet[:,:-1],trainingSet[:,-1])
            y_pred = Training.predict(testSet[:,:-1])

            #Compare the true and predicted labels and calculate the error
            y_true = testSet[:,-1]
            avErr += np.count_nonzero(y_true != y_pred) / y_true.size

    avErr = (1/(nfolds * nrepetitions))*avErr

    if retMet:
        return Training

    return avErr



  
class krr():
    ''' your header here!
    '''
    def __init__(self, kernel='linear', kernelparameter=1, regularization=0):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization

    def fit(self, X, y, kernel=False, kernelparameter=False, regularization=False):
        ''' your header here!
        '''
        if kernel is not False:
            self.kernel = kernel
        if kernelparameter is not False:
            self.kernelparameter = kernelparameter
        if regularization is not False:
            self.regularization = regularization

        #Calculate the Kernel Matrix K
        K = self.calcKer(X, kernel, kernelparameter)

        #Calculate the Regularization term C
        if regularization == 0:

            eigVals, _  = np.linalg.eig(K)
            eigMean     = np.mean(eigVals)
            paramList   = ...


        #Use Cross-Validation to find the Kernel Parameters

        #Perform the kernel ridge regression

        return self

    def predict(self, X):
        ''' your header here!
        '''
        return self


    def ker(self, x, y, ker, kernelParameter):

        # Linear Kernel
        if ker == 'linear':
            return x*y

        # Polynomial Kernel
        if ker == 'polynomial':
            d = kernelParameter
            return ((x*y)+1)**d

        # Gaussian Kernel
        if ker == 'gaussian':
            sigma = kernelParameter
            exponent = -(np.abs((x-y))**2)/(2*(sigma**2))
            return np.exp(exponent)

    def calcKer(self, X, kernel, kernelparameter):

        # Linear Kernel
        if kernel == 'linear':
            return X@X.T

        # Polynomial Kernel
        if kernel == 'polynomial':
            d = kernelparameter
            one = np.ones(X.shape)
            return np.linalg.matrix_power((X@X.T+one),d)

        # Gaussian Kernel
        if kernel == 'gaussian':
            sigma = kernelparameter
            G           = A@A.T
            g           = np.diag(G)
            one         = np.ones((g.shape[0]))
            distances   = np.outer(g,one) + np.outer(one,g) - 2*G
            params      = (-1/(sigma**2))*distances
            return np.exp(params)


