""" ps4_implementation.py

PUT YOUR NAME HERE:
<FIRST_NAME><LAST_NAME>


Complete the classes and functions
- svm_qp
- plot_svm_2d
- neural_network
Write your implementations in the given functions stubs!


(c) Felix Brockherde, TU Berlin, 2013
    Jacob Kauffmann, TU Berlin, 2019
"""
import scipy.linalg as la
import matplotlib.pyplot as plt
import sklearn.svm
from cvxopt.solvers import qp
from cvxopt import matrix as cvxmatrix
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class svm_qp():
    """ Support Vector Machines via Quadratic Programming """

    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.C = C
        self.alpha_sv = None
        self.b = None
        self.X_sv = None
        self.Y_sv = None
    
    def fit(self, X, Y):

        # INSERT_CODE
        
        # Here you have to set the matrices as in the general QP problem
        #P = 
        #q = 
        #G = 
        #h = 
        #A =   # hint: this has to be a row vector
        #b =   # hint: this has to be a scalar
        
        # this is already implemented so you don't have to
        # read throught the cvxopt manual
        alpha = np.array(qp(cvxmatrix(P, tc='d'),
                            cvxmatrix(q, tc='d'),
                            cvxmatrix(G, tc='d'),
                            cvxmatrix(h, tc='d'),
                            cvxmatrix(A, tc='d'),
                            cvxmatrix(b, tc='d'))['x']).flatten()

        #b = 

    def predict(self, X):

        # INSERT_CODE

        return self


# This is already implemented for your convenience
class svm_sklearn():
    """ SVM via scikit-learn """
    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        if kernel == 'gaussian':
            kernel = 'rbf'
        self.clf = sklearn.svm.SVC(C=C,
                                   kernel=kernel,
                                   gamma=1./(1./2. * kernelparameter ** 2),
                                   degree=kernelparameter,
                                   coef0=kernelparameter)

    def fit(self, X, y):
        self.clf.fit(X, y)
        self.X_sv = X[self.clf.support_, :]
        self.y_sv = y[self.clf.support_]

    def predict(self, X):
        return self.clf.decision_function(X)


def plot_boundary_2d(X, y, model):

    #Plot
    plt.scatter(X[:,0],X[:,1], c = y)
    plt.show()

    pass


def sqdistmat(X, Y=False):
    if Y is False:
        X2 = sum(X**2, 0)[np.newaxis, :]
        D2 = X2 + X2.T - 2*np.dot(X.T, X)
    else:
        X2 = sum(X**2, 0)[:, np.newaxis]
        Y2 = sum(Y**2, 0)[np.newaxis, :]
        D2 = X2 + Y2 - 2*np.dot(X.T, Y)
    return D2


def buildKernel(X, Y=False, kernel='linear', kernelparameter=0):
    d, n = X.shape
    if Y.isinstance(bool) and Y is False:
        Y = X
    if kernel == 'linear':
        K = np.dot(X.T, Y)
    elif kernel == 'polynomial':
        K = np.dot(X.T, Y) + 1
        K = K**kernelparameter
    elif kernel == 'gaussian':
        K = sqdistmat(X, Y)
        K = np.exp(K / (-2 * kernelparameter**2))
    else:
        raise Exception('unspecified kernel')
    return K

class neural_network(nn.Module):

    def __init__(self, layers=[2,100,2], scale=.1, p=None, lr=None, lam=None):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(scale*torch.randn(m, n)) for m, n in zip(layers[:-1], layers[1:])])
        self.biases = nn.ParameterList([nn.Parameter(scale*torch.randn(n)) for n in layers[1:]])

        self.p = p
        self.lr = lr
        self.lam = lam
        self.train = False

    def relu(self, X, W, b):

        #get the dimensions
        n       = X.shape[0]
        inDim   = X.shape[1]
        outDim  = b.shape[0]

        #Case 1: in Training
        Z = X@W+b

        if self.train == True:
            delta = self.bernouli(torch.rand(outDim))
            Z = delta*Z
            Z[Z < 0] = 0

        #Case 2: in Testing
        if self.train == False:
            Z       = (1-self.p)*Z
            Z[Z<0]  = 0

        #return the Output
        return(Z)

    def softmax(self, X, W, b):

        #Calculate Z
        Z = X @ W + b

        #Calculate the Denominator
        denom = 0
        for z in Z:
            denom += torch.exp(z)

        #Calculate and return Y
        Y = torch.exp(Z)/(denom)

        return Y

    def forward(self, X):

        X = torch.tensor(X, dtype=torch.float) #reqiures_grad=true?

        #Relu Layers - for loop
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            X = self.relu(X, weight, bias)

        #Softmax Layer
        X = self.softmax(X, self.weights[-1], self.biases[-1])

        return X

    def predict(self, X):
        return self.forward(X).detach().numpy()

    def loss(self, ypred, ytrue):

        #The loss is still too high compared to the suggested solution
        #Optimizuation works, though and the loss is proportionally right

        #Initialize the loss
        loss = 0

        #Nested for loop to compute the loss
        for predLine,trueLine in zip(ypred, ytrue):
            for p, t in zip(predLine, trueLine):
                loss += t*torch.log(p)

        loss *= -1/(ypred.shape[0])

        #Return the los
        return loss

    def fit(self, X, y, nsteps=1000, bs=100, plot=False):
        X, y = torch.tensor(X), torch.tensor(y)
        optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.lam)

        I = torch.randperm(X.shape[0])
        n = int(np.ceil(.1 * X.shape[0]))
        Xtrain, ytrain = X[I[:n]], y[I[:n]]
        Xval, yval = X[I[n:]], y[I[n:]]

        Ltrain, Lval, Aval = [], [], []
        for i in range(nsteps):
            optimizer.zero_grad()
            I = torch.randperm(Xtrain.shape[0])[:bs]
            self.train = True
            output = self.loss(self.forward(Xtrain[I]), ytrain[I])
            self.train = False
            Ltrain += [output.item()]
            output.backward()
            optimizer.step()

            outval = self.forward(Xval)
            Lval += [self.loss(outval, yval).item()]
            Aval += [np.array(outval.argmax(-1) == yval.argmax(-1)).mean()]

        if plot:
            plt.plot(range(nsteps), Ltrain, label='Training loss')
            plt.plot(range(nsteps), Lval, label='Validation loss')
            plt.plot(range(nsteps), Aval, label='Validation acc')
            plt.legend()
            plt.show()

    def bernouli(self, X):

        # Create the array to be returned
        Y = torch.zeros(X.size())

        # iterate over the array of random numbers
        for count, x in enumerate(X):

            if x > self.p:
                Y[count] = 1

        return Y

def testNN():

    nn   = neural_network()
    nn.train = False
    nn.p = 0.2

    X = torch.ones(3,3)    #torch.rand(3,3)
    X[1] = torch.tensor([1,-5, 1])
    W = 2 * torch.ones(3,3)    #torch.rand(3,3)
    b = 4 * torch.ones(3)
    ypred = torch.tensor([0.936, 0.028, 0.013, 0.023])
    ytrue = torch.tensor([1, 0, 0, 0])

    print(nn.relu(X,W,b))
    print(nn.softmax(X,W,b))
    #print(nn.loss(ypred,ytrue))
