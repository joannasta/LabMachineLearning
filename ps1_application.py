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



def app_auc(gamma, plot):
    p = (gamma.shape[0]-5300)
    y_true = np.asarray((5300*[-1] + p*[1]))
    y_pred = np.asarray(((2*gamma/np.amax(gamma))-1))
    trapez_s = imp.auc(y_true,y_pred,plot)
    return trapez_s

def sample_outliers(k):
    return np.random.uniform(-4,4,(2,int(k*5300)))

def calc_indexes(dataset,outliers):
    newdata=(np.append(dataset, outliers, axis=1)).T
    gamma3= imp.gammaidx(newdata, 3)
    gamma10= imp.gammaidx(newdata, 10)
    x=np.mean(newdata.T, axis=1)
    x=x.reshape((2,1))
    x=np.repeat(x, newdata.shape[0], axis=1)
    disToMean=np.linalg.norm((np.subtract(x,newdata.T)), axis=0)
    return gamma3, gamma10, disToMean

def outliers_calc():
    banana = np.load('./data/banana.npz')
    dataset, labels = banana["data"], banana["label"]
    k = 0.1
    rep=100
    gamma3_List, gamma10_List,disToMean_List= np.zeros((rep)), np.zeros((rep)),np.zeros((rep))
    for i in range(rep):
        #1) Sample a random set of outliers (...)
        #k ist der Anteil der generierten Outliers
        #insgesamt 5300 Werte -> 1% =  53, 10%= 530, 50%=2650, 100%=5300

        outliers = sample_outliers(k)

        #2) Add the outliers, compute g-index with k=3, k=10 distance to the mean
        gamma3,gamma10,disToMean = calc_indexes(dataset,outliers)

        # 3) compute the AUC
        gamma3_List[i]    = (app_auc(gamma3, False))
        gamma10_List[i]   = (app_auc(gamma10, False))
        disToMean_List[i] = (app_auc(disToMean, False))

    fig1, axs = plt.subplots(3)
    #axs[0].set_title('gamma3 Plot')
    axs[0].boxplot(gamma3_List)
    #axs[1].set_title('gamma10 Plot')
    axs[1].boxplot(gamma10_List)
    #axs[2].set_title('disToMean Plot')
    axs[2].boxplot(disToMean_List)
    plt.show()
    # np.savez_compressed('outliers.npz', var1=var1, var2=var2, ...)

def exemplaryPlot():
    banana = np.load('./data/banana.npz')
    dataset, labels = banana["data"], banana["label"]
    outliers = sample_outliers(0.2)
    gamma3,gamma10,disToMean = calc_indexes(dataset,outliers)
    gamma3 = (app_auc(gamma3, True))
    gamma10 = (app_auc(gamma10, True))
    disToMean = (app_auc(disToMean, True))
    plt.scatter(dataset[0,::],dataset[1,::])
    plt.scatter(outliers[0,p::],outliers[1,::])
    plt.show()



def outliers_disp():
    ''' display the boxplots'''
    # results = np.load('outliers.npz')


def lle_visualize(dataset='flatroll'):

    #1) 3D-plot

    #2) # n_rule: "knn" or "eps-ball"
        #param: Anzahl der nearest neighbours oder gew. epsilon
        #imp.lle(fish_data, fish_reference, "knn", 50, 1e-2)
        #calculate lle
    #3) plot lle-embedding

    #Hyperparameter
    n_rule="eps-ball"
    param = 10

    if dataset == "flatroll":
        flatroll = np.load('./data/flatroll_data.npz')
        data = (flatroll["Xflat"]).T
        reference = flatroll["true_embedding"]
        plt.scatter(data[:,0], data[:,1], cmap='Greens')

        LLE = imp.lle(data, reference, n_rule, param, tol=1e-2)
        plt.scatter(LLE[:,0],LLE[:,1])
        plt.show()

    if dataset == "fishbowl":
        fishbowl = np.load('./data/fishbowl_dense.npz')
        data = fishbowl["X"].T
        reference = fishbowl["X"].T[:,2]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        zdata = 15 * np.random.random(2000)
        ax.scatter3D(data[:,0], data[:,1], data[:,2], c=zdata, cmap='Greens')

        LLE = imp.lle(data, reference, n_rule, param, tol=1e-2)
        plt.scatter(LLE[:,0],LLE[:,1])
        plt.show()

    if dataset == "swissroll":
        swissroll = np.load('./data/swissroll_data.npz')
        data = (swissroll["x_noisefree"]).T
        reference = swissroll["z"].T[:,0]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        zdata = 15 * np.random.random(400)
        #ax.scatter3D(data[:,0], data[:,1], data[:,2], c=zdata, cmap='Greens')

        reference = np.zeros((400,2))
        LLE = imp.lle(data, reference, n_rule, param, tol=1e-2)
        ax.scatter(LLE[:,0],LLE[:,1], c='r')
        plt.show()
        #print(LLE[:,0],LLE[:,1].shape)

def lle_noise():
    ''' LLE under noise for assignment 8'''

def lle_noise():
    ''' LLE under noise for assignment 8'''

