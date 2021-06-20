""" ps3_application.py

PUT YOUR NAME HERE:
<FIRST_NAME><LAST_NAME>


Write the functions
- roc_curve
- krr_app
(- roc_fun)

Write your code in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import pylab as pl
import random
# import matplotlib as pl
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.lines import Line2D
from scipy.stats import norm
import scipy.io
import os
import sys
import pickle
import itertools
import matplotlib.pyplot as plt

import ps3_implementation as imp
#import ps3_implementation_application as impap
#imp = reload(imp)


def assignment_3():

    #Booleans for the sub-assignments
    a = False
    b = True
    c = True

    mat = scipy.io.loadmat('./data/qm7.mat')
    # 1) Load the USPS dataset
    X = mat['X']
    T = mat['T']

     #Get the Eigenvalues sorted in ascending order
    xEig = np.zeros((7165,23))

    for count, x in enumerate(X):
        eigVals, _  = np.linalg.eig(x)
        xEig[count] = np.sort(eigVals)


    #Assignment a)
    if (a):

        #The plot works. Stepsize one takes forever, longer stepsizes are less acurrate, but faster
        stepSize = 1
        aX = np.array([np.linalg.norm(xEig[p[0]]-xEig[p[1]]) for p in itertools.product(np.arange(7165, step = stepSize), repeat=2)])
        aY = np.array([np.abs(T[:,p[0]]-T[:,p[1]])[0] for p in itertools.product(np.arange(7165, step = stepSize), repeat=2)])
        plt.scatter(aX, a)
        plt.show()

    #Assignment b)
    if (b):

        XT          = np.array([*xEig.T, *T]).T
        np.random.shuffle(XT)
        trainSet    = XT[0:5000]
        testSet     = XT[500:-1]

    #Assignment c)
    if (c):

        #Obtain the parameter values
        aX = np.array([np.linalg.norm(xEig[p[0]] - xEig[p[1]]) for p in itertools.product(np.arange(7165, step=10), repeat=2)])
        Sigmas = np.array([np.quantile(aX, 0.1*i+0.1) for i in range(10)])
        C = np.logspace(-7, 0, num=8, base=10)
        params = [Sigmas, C]

        #Obtain the validation Data
        valSetx = trainSet[0:2500,:-1]
        valSety = trainSet[0:2500,-1]

        params = {'kernel': ['gaussian'], 'kernelparameter': Sigmas, 'regularization': C}
        cvkrr = imp.cv(valSetx, valSety, imp.krr, params, loss_function=mean_absolute_error, nfolds = 5, nrepetitions=1)
        #ypred = cvkrr.predict(Xte)
        #print('Gaussian kernel parameter: ', cvkrr.kernelparameter)
        #print('Regularization paramter: ', cvkrr.regularization)
        #print(crossValidated.kernel, "CDONE")
        print(cvkrr.kernelparameter, cvkrr.regularization, cvkrr.cvloss)


    if(d):

        for i in range(100,5000):
            return





def roc_curve(fpr,tpr):
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def roc_fun(y_true, y_pred):
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

    biases= np.arange(-0.5,0.5,0.1)
    for bias in biases:
        for i in range(y_pred.shape[0]):
            if y_pred[i]+bias >=0:
                y_pred[i] = 1
            else:
                y_pred[i] = -1
        t = (y_true[i],y_pred[i])
        if (y_true[i],y_pred[i]) == (1,1):
            TP = TP +1
        if (y_true[i],y_pred[i]) == (-1,1):
            FP = FP +1
        if (y_true[i],y_pred[i]) == (1,-1):
            FN = FN +1
        if (y_true[i],y_pred[i]) == (-1,-1):
            TN = TN +1
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        roc_curve(FPR,TPR)
        Rates.append((TPR,FPR))
    return biases[np.argmin(rates)]

    
def krr_app(reg=False):
    ''' your header here!
    '''

    banana_xtest = np.loadtxt('./data/U04_banana-xtest.dat')
    banana_xtrain = np.loadtxt('./data/U04_banana-xtrain.dat')
    banana_ytest = np.loadtxt('./data/U04_banana-ytest.dat')
    banana_ytrain = np.loadtxt('./data/U04_banana-ytrain.dat')
    params_banana = { 'kernel': ['gaussian'], 'kernelparameter': np.logspace(-4,4,20),
                      'regularization': [0]}
    print("banana_xtrain, banana_ytest",banana_xtrain.shape, banana_ytrain.shape)
    print("start banana cv")
    krr_banana = imp.cv(banana_xtrain, banana_ytrain, imp.krr, params_banana, loss_function=imp.zero_one_loss, nfolds=10, nrepetitions=5)
    print("end banana cv")
    banana_y_pred = krr_banana.predict(banana_xtest)
    result_banana={'cvloss':krr_banana.cvloss,
    'kernel' : krr_banana.kernel,
    'kernelparameter' : krr_banana.kernelparameter,
    'regularization' : krr_banana.regularization,
    'y_pred'  : banana_y_pred}


    diabetis_xtest = np.loadtxt('./data/U04_diabetis-xtest.dat')
    diabetis_xtrain = np.loadtxt('./data/U04_diabetis-xtrain.dat')
    diabetis_ytest = np.loadtxt('./data/U04_diabetis-ytest.dat')
    diabetis_ytrain = np.loadtxt('./data/U04_diabetis-ytrain.dat')
    params_diabetis = { 'kernel': ['gaussian'], 'kernelparameter': np.logspace(-4,4,20),
                      'regularization': [0]}
    print("start diabetis cv")
    krr_diabetis = imp.cv(diabetis_xtrain, diabetis_ytrain, imp.krr, params_diabetis, loss_function=imp.zero_one_loss, nfolds=13, nrepetitions=5)
    print("end diabetis cv")
    diabetis_y_pred = krr_diabetis.predict(diabetis_xtest)
    result_diabetis={'cvloss':krr_diabetis.cvloss,
    'kernel' : krr_diabetis.kernel,
    'kernelparameter' : krr_diabetis.kernelparameter,
    'regularization' : krr_diabetis.regularization,
    'y_pred'  : diabetis_y_pred}


    flare_solar_xtest = np.loadtxt('./data/U04_flare-solar-xtest.dat')
    flare_solar_xtrain = np.loadtxt('./data/U04_flare-solar-xtrain.dat')
    flare_solar_ytest = np.loadtxt('./data/U04_flare-solar-ytest.dat')
    flare_solar_ytrain = np.loadtxt('./data/U04_flare-solar-ytrain.dat')
    params_flare_solar = { 'kernel': ['gaussian'], 'kernelparameter': np.logspace(-4,4,20),
                      'regularization': [0]}
    print("start flare_solar_ cv")
    krr_flare_solar = imp.cv(flare_solar_xtrain, flare_solar_ytrain, imp.krr, params_flare_solar, loss_function=imp.zero_one_loss, nfolds=10, nrepetitions=5)
    print("end flare_solar_ cv")
    flare_solar_y_pred = krr_flare_solar.predict(flare_solar_xtest)
    result_flare_solar={'cvloss':krr_flare_solar.cvloss,
    'kernel' : krr_flare_solar.kernel,
    'kernelparameter' : krr_flare_solar.kernelparameter,
    'regularization' : krr_flare_solar.regularization,
    'y_pred'  : flare_solar_y_pred}


    image_xtest = np.loadtxt('./data/U04_image-xtest.dat')
    image_xtrain = np.loadtxt('./data/U04_image-xtrain.dat')
    image_ytest = np.loadtxt('./data/U04_image-ytest.dat')
    image_ytrain = np.loadtxt('./data/U04_image-ytrain.dat')
    params_image = { 'kernel': ['gaussian'], 'kernelparameter': np.logspace(-4,4,20),
                      'regularization': [0]}
    print("start image cv")
    krr_image = imp.cv(image_xtrain, image_ytrain, imp.krr, params_image, loss_function=imp.zero_one_loss, nfolds=10, nrepetitions=5)
    print("end image cv")
    image_y_pred = krr_image.predict(image_xtest)
    result_image={'cvloss':krr_image.cvloss,
    'kernel' : krr_image.kernel,
    'kernelparameter' : krr_image.kernelparameter,
    'regularization' : krr_image.regularization,
    'y_pred'  : image_y_pred}                  


    ringnorm_xtest = np.loadtxt('./data/U04_ringnorm-xtest.dat')
    ringnorm_xtrain = np.loadtxt('./data/U04_ringnorm-xtrain.dat')
    ringnorm_ytest = np.loadtxt('./data/U04_ringnorm-ytest.dat')
    ringnorm_ytrain = np.loadtxt('./data/U04_ringnorm-ytrain.dat')
    params_ringnorm = { 'kernel': ['gaussian'], 'kernelparameter': np.logspace(-4,4,20),
                      'regularization': [0]}
    print("start ringnorm cv")
    krr_ringnorm = imp.cv(ringnorm_xtrain, iringnorm_ytrain, imp.krr, params_ringnorm, loss_function=imp.zero_one_loss, nfolds=20, nrepetitions=5)
    print("end ringnorm cv")
    ringnorm_y_pred = krr_ringnorm.predict(ringnorm_xtest)
    result_ringnorm={'cvloss':krr_ringnorm.cvloss,
    'kernel' : krr_ringnorm.kernel,
    'kernelparameter' : krr_ringnorm.kernelparameter,
    'regularization' : krr_ringnorm.regularization,
    'y_pred'  : ringnorm_y_pred}


    results ={
    "banana":result_banana,
    "diabetis":result_diabetis,
    "flare_solar":result_flare_solar,
    "image":result_image,
    "ringnorm":result_ringnorm}


    with open('results.p', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    




    

def mean_absolute_error(y_true, y_pred):
    return (1 / y_true.shape[0]) * np.sum(np.abs(y_pred - y_true))
