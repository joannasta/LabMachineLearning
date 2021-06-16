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

    #Shufflen und die Repititions hier erstellen und an cross_validate übergeben...
    for i in it.product(*params.values()):
        met = cross_validate(X, y, method, [*i], loss_function, nfolds, nrepetitions)
        errList.append(met.cvloss)
        metList.append(met)



    if len(errList) == 1:
        return errList[0]

    argmin = np.argmin(errList)
    return metList[argmin]

#Shufflen und repetion in CV, für alle Settings die gleichen Folds


def cross_validate(X, y, method, paramList, loss_function, nfolds, nrepetitions):

    #Initialize the average error
    avErr = 0

    for repetition in range(nrepetitions):

        # Calculate the size of the partitions
        #Funktioniert das auch bei höheren Dimensionen der Daten?
        XandY = Z = np.array([*X.T,y]).T
        np.random.shuffle(XandY)
        partitions = np.array(np.array_split(XandY, nfolds, axis=0))

        tempErr = 0

        for fold in range(nfolds):

            # Create TestSet and TrainingSet
            testSet = partitions[fold]

            #Create the training set
            a = np.arange(nfolds)
            a = a[a != fold]
            trainingSet = np.array(partitions[a])
            trainingSet = trainingSet.reshape(trainingSet.shape[0]*trainingSet.shape[1], trainingSet.shape[2])

            # Train and Predict the Data
            Training = method(paramList)

            Training.fit(trainingSet[:, :-1], trainingSet[:, -1])
            #testSet = testSet.T
            y_pred = Training.predict(testSet)

            # Compare the true and predicted labels and calculate the error
            y_true = testSet[:,-1]  # [:-2]

            #Use the specified loss function from the parameters
            tempErr += np.count_nonzero(y_true != y_pred) / y_true.size
            avErr += tempErr

    avErr = (1 / (nfolds * nrepetitions)) * avErr #SCOPE OF AVERR!
    Training.cvloss = avErr

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
        self.K = self.calcKer(X, kernel, kernelparameter)

        # Calculate the Regularization term C
        if regularization == 0:
            # Calculate the mean of the Eigenvalues as the center of candidates
            eigVals, eigVecs = np.linalg.eig(self.K)
            eigMean = np.mean(eigVals)

            # Create a List of candidates
            paramList = np.logspace(-3, 4, num=10, base=eigMean)

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

        # Calculat the error
        eps = 0
        for i in range(n):
            eps += ((y[i] - Sy[i]) / (1 - S[i, i])) ** 2
        eps = eps / n

        return eps


    def predict(self, Y):

        predictions = np.zeros(Y.shape[0])

        #Man kann beide Loops vermeiden:
        #berechne die kernel matrix zwischen trainings und testdaten
        #-> nxm Matrix für n Trainings und m Testdaten
        # alpha ist m dimensionales array
        #-> dot product zw alpha und A (A.T@alpha)

        #iterate over the data points
        for i, data in enumerate(Y):

            #calculate the sum
            for j in range(self.alpha.shape[0]):

                tempKer = self.ker(data, self.X[j], self.kernel, self.kernelparameter)[0]
                predictions[i] += self.alpha[j] * tempKer

        return predictions

    def ker(self, x, y, ker, kernelParameter):
        #print(x.shape, y.shape, "ID25")
        #print(ker)
        #print(x,y,"---XY---")

        # print("x",x.shape,"y",y.shape,"ker",ker,"kernelP",kernelParameter)
        #print("KER",ker)
        if  isinstance(ker, list):
            kern = ker[0]
            sigma = ker[1]
            d = ker[2]
        else:
            kern = ker
            sigma = kernelParameter  # ker[1]
            d = kernelParameter  # ker[2]

        #
        #print("Kern, Sigma, d" , kern, sigma, d, "-------")

        # Linear Kernel
        if kern == 'linear':
            return x * y

        # Polynomial Kernel
        if kern == 'polynomial':
            return ((x * y) + 1) ** d

        # Gaussian Kernel
        if kern == 'gaussian':

            exponent = - (np.abs((x - y)) ** 2) / (2 * (sigma ** 2))
            #print("exponent",exponent)
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
            #G = A @ A.T
            G = X @ X.T
            g = np.diag(G)
            one = np.ones((g.shape[0]))
            distances = np.outer(g, one) + np.outer(one, g) - 2 * G
            params = (-1 / (sigma ** 2)) * distances
            return np.exp(params)

        # "Catch" the wrong specifications
        return X @ X.T


def centerX(self):
    av = np.zerpos(self.X.shape[0])
    for vec in self.X:
        av += vec
    av = (1 / X.shape[0]) * av
    self.X = X - av


