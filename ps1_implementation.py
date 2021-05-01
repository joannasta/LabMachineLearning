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
import matplotlib.pyplot as plt

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
        return np.cov(self.centerMatrix(Xtrain))
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
        1
        centeredMatrix=Xtrain-meanMat
        #returning the centered Data
        return centeredMatrix

    def project(self, Xtest, m):
        CTest=self.centerMatrix(Xtest)
        return (CTest@(self.U[1:m].T))
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
    n=X.shape[0]
    gamma = np.zeros ((n))
    for i in range(X.shape[0]):
        distance = np.linalg.norm( X[i] - X , axis = 1) # nx1
        distance = np.argsort(distance)[:k+1]
        gamma[i] =(1/k)*np.sum(distance)
    return gamma

def trapez(X,Y):
    s=0
    for idx,x in enumerate(X[0:-1]):
        s+= (x-X[idx+1])*(Y[idx]+0.5*(Y[idx+1]-Y[idx]))
    return s
def auc(y_true, y_pred, plot=False):
    # ...
    # you may use numpy.trapz here

    P = y_true[np.where(y_true == 1)[0]]
    N = y_true[np.where(y_true == -1)[0]]
    n = y_pred.shape[0]
    ROC = make_coordinates(y_true,y_pred,P,N)
    if plot :
        plot_ROC(ROC)
    trapez_s = trapez(ROC[::,0],ROC[::,1])
    return trapez_s

def plot_ROC(C):
    plt.step(C[:,0],C[:,1])
    plt.show()


def make_coordinates(y_true,y_pred,P,N):
    C   = np.zeros((y_true.shape[0]+1,2))
    idx = np.argsort(y_pred)
    Columns_to_del = []
    y_true_sorted = y_true[idx]
    y_pred_sorted = y_pred[idx]
    C[0,0] = 1
    C[0,1] = 1
    for i,p in enumerate(y_pred_sorted):
        if i != y_pred_sorted.shape[0]-1 and   p == y_pred_sorted[i+1]:
            Columns_to_del.append(i+1)
            C[i+1,0] = -1
            C[i+1,1] = -1

        else:
            y_true_sliced = y_true_sorted[i+1::]

            TP = (y_true_sliced[np.where(y_true_sliced == 1)[0]]).shape[0]
            FP = (y_true_sliced[np.where(y_true_sliced == -1)[0]]).shape[0]
            # TPR
            TPR = TP/P.shape[0]
            # FPR
            FPR = FP/N.shape[0]
            C[i+1,0] = FPR
            C[i+1,1] = TPR


    C = np.delete(C,Columns_to_del,axis=0)

    return C





def lle(X, m, n_rule, param, tol=1e-2):
    #print("m", m.shape, "X", X.shape)
    # Input X dxn matrix
    #Hyperparameter: n_rule, param, tol=1e-2
    print ('Step 1: Finding the nearest neighbours by rule ' + n_rule)
    try:
        if n_rule == "knn":
            #print("knn-if")
            Neighbors = (k_nearest_neighbor(X,param))
        elif n_rule == "eps-ball":
            Neighbors = (eps_ball(X,param))
    except :
        raise ValueError("keine gueltige Eingabe")
    #print("Neighbors", Neighbors.shape)

    # ...
    print ('Step 2: local reconstruction weights')
    n = X.shape[0]
    W = np.zeros((n,n))
    for i in range(n):
        # hier nochmal rübergucken
        k=0
        if n_rule=="knn":
            k = Neighbors[i].shape[0]
        else:
            k= int(Neighbors[i,n-1])
        C = np.zeros((k,k))
        for j in range(k):
            for l in range(j,k,1):
                #print("X[i]", X[i])
                #print("neighbours", Neighbors[i,j])
                #print("X[Neighbors[i,j]]).T", X[int(Neighbors[i,j])].T)
                #print("X[Neighbors[i,l]]", X[(int(Neighbors[i,l]))])
                C[j,l] = (X[i] - X[int(Neighbors[i,j])]).T@(X[i]-X[int(Neighbors[i,l])])
                C[l,j] = C[j,l]
        w = np.linalg.inv(C + tol*np.eye(k))@np.ones((k))
        w = (1/(np.dot(w,np.ones(k))))*w
        for o in range(k):
            #print("W", W.shape, "w", w.shape, k)
            #eventuell andere symmetrie beachten?
            eta = int(Neighbors[i,o])
            W[i,eta] = w[o]

    n=X.shape[0]
    M = np.zeros((n,n))
    for i in range (n):
        for j in range(n):
            k=0
            if i == j :
                k = 1
                # Zeile 22 in Pseudocode abgleichen
                #eventuell ist die Transponierte notwendig
            M[i,j] = k - W[i,j] - W[j,i] + np.sum(W[i]@W[j], axis=0)
    #print(np.sum(M))
    # ...
    print ('Step 3: compute embedding')
    #print(m.shape)
    #print(m.size)
    #print(M.shape)


    d = int(m.size / X.shape[0])
    #print(d)
    #folgende zeile vlt auskommentieren
    #n=X.shape[0]
    Z = np.zeros((n,d))
    #print(Z.shape)
    eigenValues , eigenVectors = np.linalg.eig(M)
    order=eigenValues.argsort()
    eigenVectors = eigenVectors[:,order]
    #Z=
    #print("EV-Mat", eigenVectors.shape)
    #print(eigenVectors.shape)
    #print(d)
    for i in range(n):
        #print(eigenVectors[i,1:d+1])
        #print("i")
        #print(eigenVectors[i,1:d+1], eigenVectors[i,1:d+1].shape)
        Z[i]=eigenVectors[i,1:d+1]
        #print(i, (eigenVectors.T[i][2:d+1]).shape,d)
        #Z[i]=eigenVectors.T[i][2:d+1]
        """for j in range(i+1,d+1,1):
            Z[i]=eigenVectors[i,j]
            #np.append(Z[i-1], eigenVectors[i,j])"""
    return (Z)

    # ...

def eps_ball(X,epsilon):
    # nxd
    n = X.shape[0]
    Distances_idx=np.zeros((n,n))
    for i in range(X.shape[0]):
        count=0
        #Distances_idx.append([])
        for j in range(X.shape[0]):
            if i != j :
               if np.linalg.norm(X[i]-X[j]) < epsilon:
                    Distances_idx[i,count]=j
                    count+=1
        Distances_idx[i,n-1]=count
    return Distances_idx


def k_nearest_neighbor(X,k):
    Neighbors = np.zeros((X.shape[0],k))
    for i in range(X.shape[0]):
        Y = np.delete(X,(i),axis=0)
        distance = np.linalg.norm( X[i] - Y, axis = 1) # nx1
        distance = np.argsort(distance)[:k]
        #print("Neighbors", Neighbors.shape, "distance", distance.shape)
        Neighbors[i] = distance
    return Neighbors

        return np.cov(self.centerMatrix(Xtrain))
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
        1
        centeredMatrix=Xtrain-meanMat
        #returning the centered Data
        return centeredMatrix

    def project(self, Xtest, m):
        CTest=self.centerMatrix(Xtest)
        return (CTest@(self.U[1:m].T))
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
    n=X.shape[0]
    gamma = np.zeros ((n))
    for i in range(X.shape[0]):
        distance = np.linalg.norm( X[i] - X , axis = 1) # nx1
        distance = np.argsort(distance)[:k+1]
        gamma[i] =(1/k)*np.sum(distance)
    return gamma

def trapez(X,Y):
    s=0
    for idx,x in enumerate(X[0:-1]):
        s+= (x-X[idx+1])*(Y[idx]+0.5*(Y[idx+1]-Y[idx]))
    return s
def auc(y_true, y_pred, plot=False):
    # ...
    # you may use numpy.trapz here

    P = y_true[np.where(y_true == 1)[0]]
    N = y_true[np.where(y_true == -1)[0]]
    n = y_pred.shape[0]
    ROC = make_coordinates(y_true,y_pred,P,N)
    if plot :
        plot_ROC(ROC)
    trapez_s = trapez(ROC[::,0],ROC[::,1])
    return trapez_s

def plot_ROC(C):
    plt.step(C[:,0],C[:,1])
    plt.show()


def make_coordinates(y_true,y_pred,P,N):
    C   = np.zeros((y_true.shape[0]+1,2))
    idx = np.argsort(y_pred)
    Columns_to_del = []
    y_true_sorted = y_true[idx]
    y_pred_sorted = y_pred[idx]
    C[0,0] = 1
    C[0,1] = 1
    for i,p in enumerate(y_pred_sorted):
        if i != y_pred_sorted.shape[0]-1 and   p == y_pred_sorted[i+1]:
            Columns_to_del.append(i+1)
            C[i+1,0] = -1
            C[i+1,1] = -1

        else:
            y_true_sliced = y_true_sorted[i+1::]

            TP = (y_true_sliced[np.where(y_true_sliced == 1)[0]]).shape[0]
            FP = (y_true_sliced[np.where(y_true_sliced == -1)[0]]).shape[0]
            # TPR
            TPR = TP/P.shape[0]
            # FPR
            FPR = FP/N.shape[0]
            C[i+1,0] = FPR
            C[i+1,1] = TPR


    C = np.delete(C,Columns_to_del,axis=0)

    return C





def lle(X, m, n_rule, param, tol=1e-2):
    # Input X dxn matrix

    print ('Step 1: Finding the nearest neighbours by rule ' + n_rule)
    try:
        if n_rule == "knn":
            Neighbors = k_nearest_neighbor(X,param)
        if n_rule == "eps-ball":
            Neighbors = eps_ball(X,param)
    except :
        raise ValueError("keine gueltige Eingabe")

    # ...
    print ('Step 2: local reconstruction weights')
    n = X.shape[0]
    W = np.zeros((n,n))
    for i in range(n):
        # hier nochmal rübergucken
        k = Neighbors[i].shape[0]
        C = np.zeros((k,k))
        for j in range(k):
            for l in range(j,k,1):
                C[j,l] = (X[i] - X[Neighbors[i,j]]).T@(X[i]-X[Neighbors[i,l]])
                C[l,j] = C[j,l]
        w = np.inv(C + tol*np.eye(C.shape[0]))@np.ones((C.shape[0]))
        w = (1/(w.T@np.ones(C.shape[0])))@w
        for n in range(k):
            W[n] = w
    M = np.zeros((n,n))
    for i in range (n):
        for j in range(n):
            k=0
            if i == j :
                k = 1
            M[i,j] = k - W[i,j] - W[j,i] + np.sum(W[i]@W[j], axis=0)
    # ...
    print ('Step 3: compute embedding')
    Z = np.zeros((n,m))
    eigenValues , veigenVectors = np.linalg.eig(M)
    order=eigenValues.argsort()
    eigenVectors = eigenVectors[:,order]
    for i in range(n):
        for j in range(i+1,m+1,1):
            np.append(Z[i], V[i,j])
    return Z

    # ...

def eps_ball(X,epsilon):
    # nxd
    Distances_idx =[]
    for i in range(X.shape[0]):
        Distances_idx.append([])
        for j in range(X.shape[0]):
            if i != j :
               if np.linalg.norm(X[i]-X[j]) < epsilon:
                    Distances_idx[i].append(j)
    Distances_idx = np.asarray(Distances_idx)
    return Distances_idx


def k_nearest_neighbor(X,k):
    Neighbors = np.zeros((X.shape[0],k))
    for i in range(X.shape[0]):
        Y = np.delte(X,(i),axis=0)
        distance = np.linalg.norm( X[i] - Y, axis = 0) # nx1
        distance = np.argsort(distance)[:k]
        Neighbors[i] = distance
    return Neighbors
