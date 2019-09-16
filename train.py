	# USAGE
# python train.py --save-model 1 --weights output/lenet_weights.hdf5 -p 48 -n 5
# python train.py --load-model 1 --weights output/lenet_weights.hdf5 --save-model 1 -p 48 -n 5

# import the necessary packages
from pyimagesearch.cnn.networks import LeNet
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import argparse
import cv2
import cPickle as pickle
import gzip
import sys
from pandas import DataFrame as df
from model import *

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


print 'loading data'
training_data = load_data_shared("data/kalyan.pkl.gz")

td , ty = training_data
print ty.shape

dataset_data = td
dataset_target = ty
del td
del ty
pixels = args["pixels"];
print "Size is" ,pixels,'x',pixels

print dataset_data.shape
data = dataset_data.reshape(( dataset_data.shape[0], pixels, pixels))
data = data[:, :, :, np.newaxis]
print data.shape
target = dataset_target.reshape((dataset_target.shape[0], pixels, pixels))
target = target[:,:,:,np.newaxis]
print target.shape
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data, target, test_size=0.27)
del dataset_target
del dataset_data
del data
del target

def expert_loss_bin(x, y):
	inside = K.sum(x*y)
	outside = K.sum((1-x)*y)
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


print("[INFO] compiling model...")
model = getModel(weightsPath=args["weights"] if args["load_model"] > 0 else None)

print("Train data shape")
print(trainData.shape)

if args["save_model"] > 0:
	print("[INFO] training...")
	model.fit(trainData, trainLabels, batch_size=10, nb_epoch=args["epochs"],
		verbose=1)

	# show the accuracy on the testing set
	print("[INFO] evaluating...")
	metrics = model.evaluate(testData, testLabels,
		batch_size=10, verbose=1)
	print(metrics)

if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)

import matplotlib.pyplot as plt
size_x = size_y = pixels
def plotImg(image, imgName='', isProbs = 1):
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

for i in range(len(testData)):
	probs = model.predict(testData[np.newaxis, i])
	print "plotting "+ str(i)
	plotImg(testData[i],"img/R/IMG_"+str(i)+".png", 0)
	plotImg(testLabels[i],"img/R/IMG_"+str(i)+"A.png", 0)
	plotImg(probs[0],"img/R/IMG_"+str(i)+"R.png", 1)
