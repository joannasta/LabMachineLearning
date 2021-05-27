import ps2_implementation as imp
import numpy as np
import scipy.io
import matplotlib.pyplot as pltv

#assignment 9
def usps():
    mat = scipy.io.loadmat('./data/usps.mat')
    #1) Load the USPS dataset
    L=mat.get("data_labels")
    P=mat.get("data_patterns")
    imp.kmeans(P,10)
    return 0
#assignment 7

def gaussians_5():
	#gaussians5 = np.loadtxt("._5_gaussians.npy", delimiter=',')
	gaussians_5 = np.load('./data/5_gaussians.npy',allow_pickle=True)
	#print(gaussians_5,gaussians_5.shape)
	for k in range(2,8,1):
		print("k = ",k)
		imp.kmeans(gaussians_5.T,k)
	return 0
#assignment 8
def gaussians_2():
	#gaussians2 = np.loadtxt("._2_gaussians.npy", delimiter=',')
	data = np.load('2_gaussians.npy',allow_pickle=True)
	print("data", data.shape)
	#for k in range(2,8,1):
	#	print("k = ",k)
	#	imp.kmeans(data.T,k)
	imp.kmeans(data.T,7)
	return 0
#assignment10

def lab_data():
	data = np.load('./data/lab_data.npz')
	X = data["X"]
	Y = data["Y"] # if data is outlier -1 inlier +1
