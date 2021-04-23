""" sheet1_implementation.py

Viola-Joanna Stamer, 383280
Friedrich Christian Wicke, 403336


Write the functions
- pca
- gammaidx
- lle
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
    Jacob Kauffmann, TU Berlin, 2021
"""
import numpy as np
import scipy.linalg as la
import numpy.linalg as linalg


class PCA():
    def __init__(self, Xtrain):
        # ...
        self.C=self.centerOfData(Xtrain)
        self.Cov=self.computeCovariance(Xtrain)
        self.U, self.D =self.getUandD()
        #self.D=self.getD()
        pass

    def getUandD(self):
        eigenValues, eigenVectors = np.linalg.eig(self.Cov)
        order=eigenValues.argsort()[::-1]
        return eigenVectors[:,order], eigenValues[order]

    """def getD(self):
        eigenValues, eigenVectors = np.linalg.eig(self.Cov)
        order=eigenValues.argsort()[::-1]
        return eigenValues[order]"""

    def computeCovariance(self, Xtrain):
        return np.cov(centerMatrix(Xtrain))
        """n=len(Xtrain)
        d=len(Xtrain[0])
        finalMat=np.zeros((d,d))
        for i in range(n):
            #print(Xtrain[i])
            fac2=([Xtrain[i]-self.C])
            fac1=np.transpose(fac2)
            finalMat+=np.matmul(fac1,fac2)
        Cov=(1/(n-1))*finalMat
        return Cov"""

    #This function centers the Data, which consists of vectors as rows
    #It is used to compute the covariance Matrix
    def centerOfData(self, Xtrain):
        #computing the average of the vectors
        means=np.average(Xtrain, axis=0)
        return means



    def centerMatrix(self,Xtrain):
        means=self.centerOfData(Xtrain)
        n=len(Xtrain)
        #Subtracting the average from each vector to obtain centered data
        meanMat=np.array([means]*n)
        centeredMatrix=Xtrain-meanMat
        #returning the centered Data
        return centeredMatrix

    def project(self, Xtest, m):
        CTest=centerMatrix(XTest)
        return (CTest@(U[1:m].T))
        #Input: TestData as (nxd)
        #Output: projected data in (nxm) Matrix
        #Use the m first columns of U
        #returns projection into subspace with fewer dimensions

    def denoise(self, Xtest, m):
        Z=self.project(Xtest,m)
        d=Xtest.shape[0]
        n=Z.shape[0]
        X=np.zeros((n,d))
        X=np.mean(Xtest,axis=0)+np.sum(Z@U,axis=0)
        return X


def gammaidx(X, k):
    # Input : X  is a data-matrix (nxd)
    # Output : gamma  is a vector which contains the gamma-index for each datapoint (nx1)

    gamma = np.zeros ((n))
    for i in range(X.shape[0]):
        distance = np.linalg.norm( X[i] - X , axis = 0) # nx1
        distance = np.argsort(distance)[:k+1]
        gamma[i] =(1/k)*np.sum(distance)
    return gamma


def auc(y_true, y_pred, plot=False):
    # ...
    # you may use numpy.trapz here
    pass


def lle(X, m, n_rule, param, tol=1e-2):
    print ('Step 1: Finding the nearest neighbours by rule ' + n_rule)
    # ...
    print ('Step 2: local reconstruction weights')
    # ...
    print ('Step 3: compute embedding')
    # ...
