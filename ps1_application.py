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
import networkx as nx

def usps():
    mat = scipy.io.loadmat('./data/usps.mat')
    #1) Load the USPS dataset
    L=mat.get("data_labels")
    P=mat.get("data_patterns")
    print(P.shape, L.shape)
    X=P.T
    X = X[:10].reshape((16,16,10,1))
    X = X.reshape(160,16)
    print(X.shape)
    #X = X.transpose((0,2,1,3))
    plt.imshow(X, cmap='inferno_r')
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
    k = [0.01,0.1,0.5,1.0]
    rep=100
    gamma3_List_01, gamma10_List_01,disToMean_List_01= np.zeros((rep)), np.zeros((rep)),np.zeros((rep))
    gamma3_List_10, gamma10_List_10,disToMean_List_10= np.zeros((rep)), np.zeros((rep)),np.zeros((rep))
    gamma3_List_50, gamma10_List_50,disToMean_List_50= np.zeros((rep)), np.zeros((rep)),np.zeros((rep))
    gamma3_List_100, gamma10_List_100,disToMean_List_100= np.zeros((rep)), np.zeros((rep)),np.zeros((rep))

    print("entering for-loop")

    for i in range(rep):
        #1) Sample a random set of outliers (...)
        #k ist der Anteil der generierten Outliers
        #insgesamt 5300 Werte -> 1% =  53, 10%= 530, 50%=2650, 100%=5300

        print("i =",i)
        outliers01 = sample_outliers(k[0])
        outliers10 = sample_outliers(k[1])
        outliers50 = sample_outliers(k[2])
        outliers100 = sample_outliers(k[3])
     


        #2) Add the outliers, compute g-index with k=3, k=10 distance to the mean
        gamma3_01,gamma10_01,disToMean_01 = calc_indexes(dataset,outliers01)
        gamma3_10,gamma10_10,disToMean_10 = calc_indexes(dataset,outliers10)
        gamma3_50,gamma10_50,disToMean_50 = calc_indexes(dataset,outliers50)
        gamma3_100,gamma10_100,disToMean_100 = calc_indexes(dataset,outliers100)

        # 3) compute the AUC
        gamma3_List_01[i] ,gamma10_List_01[i] ,disToMean_List_01[i]  = (app_auc(gamma3_01, False)),(app_auc(gamma10_01, False)) ,(app_auc(disToMean_01, False))
        gamma3_List_10[i] ,gamma10_List_10[i] ,disToMean_List_10[i]  = (app_auc(gamma3_10, False)),(app_auc(gamma10_10, False)) ,(app_auc(disToMean_10, False))
        gamma3_List_50[i] ,gamma10_List_50[i] ,disToMean_List_50[i]  = (app_auc(gamma3_50, False)),(app_auc(gamma10_50, False)) ,(app_auc(disToMean_50, False))
        gamma3_List_100[i] ,gamma10_List_100[i] ,disToMean_List_100[i]  = (app_auc(gamma3_100, False)),(app_auc(gamma10_100, False)) ,(app_auc(disToMean_100, False))


    print("preparing plots")
    
    gamma3 = [gamma3_List_01,gamma3_List_10,gamma3_List_50,gamma3_List_100]
    gamma10 = [gamma10_List_01,gamma10_List_10,gamma10_List_50,gamma10_List_100]
    distToMean = [disToMean_List_01,disToMean_List_10,disToMean_List_50,disToMean_List_100]

    labels = ['0.01', '0.1', '0.5','1.0']
    fig, (ax1, ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(9, 4))
    # rectangular box plot
    bplot1 = ax1.boxplot(gamma3,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
    ax1.set_title('gamma3')

    # notch shape box plot
    bplot2 = ax2.boxplot(gamma10,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
    ax2.set_title('gamma10')

    bplot3 = ax3.boxplot(distToMean,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
    ax3.set_title('Distance to Mean')
    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen']
    for bplot in (bplot1, bplot2,bplot3):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    # adding horizontal grid lines
    for ax in [ax1, ax2,ax3]:
        ax.yaxis.grid(True)
        ax.set_xlabel('contamination rates')
        #ax.set_ylabel('Observed values')

    plt.show()
   
    # np.savez_compressed('outliers.npz', var1=var1, var2=var2, ...)

def exemplaryPlot():
    banana = np.load('./data/banana.npz')
    dataset, labels = banana["data"], banana["label"]
    outliers = sample_outliers(0.2)
    gamma3,gamma10,disToMean = calc_indexes(dataset,outliers)
    gamma3 = (app_auc(gamma3, False))
    gamma10 = (app_auc(gamma10, False))
    disToMean = (app_auc(disToMean, False))
    plt.scatter(dataset[0,::],dataset[1,::])
    plt.scatter(outliers[0,::],outliers[1,::])
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
        print(reference.shape)
        #plt.scatter(data[:,0], data[:,1], cmap='Greens')
        LLE = imp.lle(data, reference, n_rule, param, tol=1e-2)
        fig, ax = plt.subplots()
        #Y = np.ones((1000,1))
        scatter = ax.scatter(LLE[:,0],reference[0,:])
        legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
        ax.add_artist(legend1)
        plt.legend()
        plt.show()

    if dataset == "fishbowl":
        fishbowl = np.load('./data/fishbowl_dense.npz')
        data = fishbowl["X"].T
        reference = fishbowl["X"].T[:,2]
        print(reference.shape)
        #fig = plt.figure()
        #ax = plt.axes(projection='3d')
        #zdata = 15 * np.random.random(2000)
        #ax.scatter3D(data[:,0], data[:,1], data[:,2], c=zdata, cmap='Greens')
        LLE = imp.lle(data, reference, n_rule, param, tol=1e-2)
        fig, ax = plt.subplots()
        ax.scatter(LLE[:,0],LLE[:,1],label="LLE-embedding")
        ax.scatter(reference[:,0],reference[:,1],label="Reference-data")
        plt.legend()
        plt.show()

    if dataset == "swissroll":
        swissroll = np.load('./data/swissroll_data.npz')
        data = (swissroll["x_noisefree"]).T
        reference = swissroll["z"].T[:,0]
        print(reference.shape)
        #fig = plt.figure()
        #ax = plt.axes(projection='3d')
        #zdata = 15 * np.random.random(400)
        #ax.scatter3D(data[:,0], data[:,1], data[:,2], c=zdata, cmap='Greens')


        reference = np.zeros((400,2))
        LLE = imp.lle(data, reference, n_rule, param, tol=1e-2)
        fig, ax = plt.subplots()
        ax.scatter(LLE[:,0],LLE[:,1], c='r',label="LLE-embedding")
        #ax.scatter(reference[:,0],reference[:,1])
        ax.scatter(LLE[:,0],reference[:,0],label="reference with LLE-embedding")
        #legend1 = ax.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
        #ax.add_artist(legend1)
        plt.legend()
        plt.show()
        #print(LLE[:,0],LLE[:,1].shape)

def lle_noise_08():
    ''' LLE under noise for assignment 8'''
    flatroll = np.load('./data/flatroll_data.npz')
    data = (flatroll["Xflat"]).T
    reference = flatroll["true_embedding"]
    noise = np.random.normal(0,0.2,(1000,2))
    data = data+noise
    param = 1
    LLE = imp.lle(data, reference,"knn",param,1e-2)
    Y = np.ones((1000,1))
    plt.scatter(LLE,Y,label=param)
    plt.legend()
    #M = nx.from_numpy_matrix(imp.adjacency_matrix(data,param))
    #data = np.array([[1,2],[6,3],[3,4],[6,7],[3,7],[3,9]])
    #M= nx.from_numpy_matrix(imp.adjacency_matrix(data,param))
    #print(imp.k_nearest_neighbor(data,1))
    #nx.draw(M, with_labels=True)
    plt.show()


def lle_noise_18():
    ''' LLE under noise for assignment 8'''
    flatroll = np.load('./data/flatroll_data.npz')
    data = (flatroll["Xflat"]).T
    N=data.shape[0]
    reference = flatroll["true_embedding"]
    noise = np.random.normal(0,1.8,(1000,2))
    #data = data+noise
    param=20
    LLE = imp.lle(data, reference,"knn",param,1e-2)
    Y = np.ones((1000,1))
    #plt.scatter(LLE,Y,label=param)
    #plt.legend()
    #M = nx.from_numpy_matrix(imp.adjacency_matrix(data,param))
    #data = np.array([[1,2],[6,3],[3,4],[6,7],[3,7],[3,9]])
    #M= nx.from_numpy_matrix(imp.adjacency_matrix(data,param))
    #print(imp.k_nearest_neighbor(data,1))
    #nx.draw(M, with_labels=True)

    #NEIGHBORHOOD Graph

    #Code inspired and partially copied from
    #https://stackoverflow.com/questions/50040310/efficient-way-to-connect-the-k-nearest-neighbors-in-a-scatterplot-using-matplotl

    #DEFINE THE NEIGHBORHOOD PARAMETER
    k=80

    #0) GET THE NEIGHBORHOOD Matrix
    neighbors = imp.k_nearest_neighbor(data, k)
    #neighbors = imp.ebs_ball(data, k)

    print(neighbors[0,0])
    #1) GET EDGE COORDINATES
    coordinates = np.zeros((N, k, 2, 2))
    for i in np.arange(N):
        for j in np.arange(k):
            coordinates[i, j, :, 0] = np.array([data[i,:][0], data[int(neighbors[i, j]), :][0]])
            coordinates[i, j, :, 1] = np.array([data[i,:][1], data[int(neighbors[i, j]), :][1]])

    #2) create line artists
    lines = LineCollection(coordinates.reshape((N*k, 2, 2)), color='black')

    fig, ax = plt.subplots(1,1,figsize = (8, 8))
    ax.scatter(data[:,0], data[:,1], c = 'black')
    ax.add_artist(lines)
    plt.show()

    #plt.scatter(data[:,0],data[:,1])
    #plt.show()
