	# USAGE
# python lenet_mnist.py --save-model 1 --weights output/lenet_weights.hdf5
# python lenet_mnist.py --load-model 1 --weights output/lenet_weights.hdf5

# import the necessary packages
from pyimagesearch.cnn.networks import LeNet
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import argparse
from scipy.ndimage.filters import gaussian_filter
import cv2
import cPickle as pickle
import gzip
import sys
from pandas import DataFrame as df
from model import *
from PIL import Image
import os, os.path

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
ap.add_argument("-p", "--pixels", type=int, default=30,
	help="size ")
ap.add_argument("-n", "--epochs", type=int, default=30,
	help="size ")
args = vars(ap.parse_args())

def load_data_shared(filename="data/mnist.pkl.gz"):
	f = gzip.open(filename, 'rb')
	training_data = pickle.load(f)
	f.close()
	return training_data


def dispI(image):
	h,w,z = 0,0,0
	invalid = 0
	if(len(image.shape)==3):
		h,w,z = image.shape
	elif len(image.shape)==2:
		h,w = image.shape
	else:
		print "INVALID IMAGE"
		invalid = 1
	x = np.amax(image)
	if(x==0):
		x = 1
	bmax  = str(np.amax(image))
	bmin = str(np.amin(image))
	image =  (image*255/x).astype("uint8")
	image = cv2.merge([image] * 3)
	if invalid == 0:
		shp = (int(600*w/(h+1))+70,600)
	else:
		shp = (600,600)
	image = cv2.resize(image, shp , interpolation=cv2.INTER_NEAREST)
	cv2.imshow("image "+bmin+" "+bmax, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

pixels = args["pixels"];

# for i in range(10):
# 	print 'data',df(trainData[5,:,:,0])
# 	print 'labels',df(trainLabels[5,:,:,0])

#for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
	#dispI(trainLabels[i].reshape(80,80))
#str_log = "1)  "
# true & pred
def expert_loss_bin(x, y):
	#print K.backend.shape(x), K.backend.shape(y)
	##str_log = str_log + "\n\n\nX-Y = " +str(x-y) +"\n";
	#str_log = str_log+ "\n\n\nX = "  + str(np.array(x))+"\n"
	#str_log = str_log + "\n\n\nY = " + str(np.array(y))+"\n"
	#str_log = str_log + "Y = "+ str(y)+"\n";
	#x = np.round(x/0.3)*0.3
	inside = K.sum(x*y)
	outside = K.sum((1-x)*y)
	#diff2 = K.sum((1-20*x)*y)/20   
	#diff2 =  K.mean(K.sum((1-x)*y, axis=-1))  -  K.mean(K.sum(x*y, axis=-1))
	#print "diff", diff2 
	return (outside) + (1-inside)

def correct_points(x, y):
	return K.sum(x*y)/K.sum(x)

def wrong_points(x, y):
	return K.sum((1-x)*y)/K.sum(1-x)

def getModel(weightsPath = None):
	input_img = Input((pixels, pixels, 1), name='img')
	model = get_unet(input_img, n_filters=4, dropout=0.05, batchnorm=True)
	if weightsPath is not None:
			print "Available:",weightsPath
			model.load_weights(weightsPath)
	model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
	model.summary()
	return model

model = getModel(weightsPath=args["weights"] if args["load_model"] > 0 else None)


import matplotlib.pyplot as plt
size_x = size_y = pixels
def plotImg(image, imgName='', isProbs = 1, threshold = 180):
	if 1==0:
		return
	h,w,z = 0,0,0
	invalid = 0
	if(len(image.shape)==3):
		h,w,z = image.shape
	elif len(image.shape)==2:
		h,w = image.shape
	else:
		print "INVALID IMAGE"
		invalid = 1
	x = np.amax(image)
	if(x==0):
		x = 1
	bmax  = str(np.amax(image))
	bmin = str(np.amin(image))
	image =  (image*255/x).astype("uint8")
	# if isProbs == 1:
	# 	super_threshold_indices = image < threshold
	# 	image[super_threshold_indices] = 0
	# 	super_threshold_indices = image >= threshold
	# 	image[super_threshold_indices] = 250
	#image = np.round(image/150)*150
	H = np.matrix(image)
	dpi = 25
	figsize = size_x/float(dpi), size_y/float(dpi)
	fig = plt.figure(figsize=figsize)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	plt.imshow(H,interpolation='none', aspect='auto')
	plt.savefig(imgName)
	#plt.show()

if (args["save_model"]>0):
	exit(0)

imgs = []
path = "spectograms"
valid_images = [".jpg",".gif",".png",".tga"]
for f in os.listdir(path):
	ext = os.path.splitext(f)[1]
	if ext.lower() not in valid_images:
		continue
	img = (np.array(Image.open(os.path.join(path,f)))/255.0)[:,:,0]
	print( img[:,:,np.newaxis].shape)
	imgs.append(   img[:,:,np.newaxis]	  ) 
	print("Added")

print(np.array(imgs).shape)
testData = np.array(imgs)

for i in range(len(testData)):
	probs = model.predict(testData[np.newaxis, i])
	#print "Final shape",probs.shape, trainData[i].shape
	print "plotting "+ str(i)
	plotImg(testData[i],"img/R/IMG_"+str(i)+".png", 0)
	#plotImg(testLabels[i],"img/R/IMG_"+str(i)+"A.png", 0)
	#dispI(trainLabels[i])
	fil = (probs[0]>0.6)*1.0
	fil = (gaussian_filter(fil, sigma=2)>0.2)*1.0
	plotImg(fil,"img/R/IMG_"+str(i)+"R.png", 1, threshold = 150)

	mit= testData[i]*(1-fil);
	plotImg(mit + gaussian_filter(mit, sigma=12)*(fil),"img/R/IMG_"+str(i)+"Diff.png", 1, threshold = 150)
