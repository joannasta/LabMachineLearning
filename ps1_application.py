""" sheet1_implementation.py

Viola-Joanna Stamer, 383280
Friedrich Christian Wicke, 403336

Write the functions
- usps
- outliers
- lle
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
    Jacob Kauffmann, TU Berlin, 2021
"""
import numpy as np
import ps1_implementation as imp
import scipy.io
import matplotlib.pyplot as plt

def usps():
    mat = scipy.io.loadmat('./data/usps.mat')
    #1) Load the USPS dataset
    L=mat.get("data_labels")
    P=mat.get("data_patterns")
    print(P.shape, L.shape)
    #print(P)

    #2a) Perform the PCA
    PCA = imp.PCA(P)
    PrincDir = PCA.U
    PrincVal = PCA.D

    #2b): visualization


    #2b a): Principal values

    #plt.plot(PrincVal)


    #2b b) 25 largest principal values as bar plots

    barvalues=["λ1","λ2","λ3","λ4","λ5","λ6","λ7","λ8","λ9","λ10","λ11","λ12","λ13","λ14","λ15","λ16","λ17","λ18","λ19","λ20","λ21","λ22","λ23","λ24","λ25"]
    #plt.bar(barvalues, PrincVal[:25].tolist())
    #print(PrincVal[:25].shape, len(barvalues))

    #2b c) 5 first principal directions as images
    #reconstructed1 = PCA.project(PrincDir[:5],5)

    #plt.imshow(P)

    #Create three scenarios:
    #lownoise, highnoise, outliers

    #3a)


    #3b)

    #3c)

    plt.show()


def outliers_calc():
    banana = np.load('./data/banana.npz')
    dataset, labels = banana["data"], banana["label"]

    #1) Sample a random set of outliers (...)
    #k ist die Anzahl der generierten Outliers
    #insgesamt 5300 Werte -> 1% =  53, 10%= 530, 50%=2650, 100%=5300
    k=53
    outliers = np.random.uniform(-4,4,(2,k))

    #2) Add the outliers, compute g-index with k=3, k=10 distance to the mean
    newdata=(np.append(dataset, outliers, axis=1)).T
    gamma3= imp.gammaidx(newdata, 3)
    gamma10= imp.gammaidx(newdata, 10)
    x=np.mean(newdata.T, axis=1)
    x=x.reshape((2,1))
    x=np.repeat(x, newdata.shape[0], axis=1)
    disToMean=np.linalg.norm((np.subtract(x,newdata.T)), axis=0)
    print(disToMean)

    # np.savez_compressed('outliers.npz', var1=var1, var2=var2, ...)


def outliers_disp():
    ''' display the boxplots'''
    # results = np.load('outliers.npz')


def lle_visualize(dataset='flatroll'):
    ''' visualization of LLE for assignment 7'''


def lle_noise():
    ''' LLE under noise for assignment 8'''

def lle_noise():
    ''' LLE under noise for assignment 8'''
