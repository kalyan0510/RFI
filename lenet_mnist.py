# USAGE
# python lenet_mnist.py --save-model 1 --weights output/lenet_weights.hdf5
# python lenet_mnist.py --load-model 1 --weights output/lenet_weights.hdf5

# import the necessary packages
import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32,cuda.root=/usr/local/cuda-8.0/bin"
from pyimagesearch.cnn.networks import LeNet
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import argparse
import cv2
import theano
import theano.tensor as T
import cPickle
import gzip
def load_data_shared(filename="data/mnist.pkl.gz"):
	f = gzip.open(filename, 'rb')
	training_data, validation_data, test_data = cPickle.load(f)
	f.close()
	def shared(data):
		"""Place the data into shared variables.  This allows Theano to copy
		the data to the GPU, if one is available.

		"""
		shared_x = theano.shared(
			np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
		shared_y = theano.shared(
			np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
		return shared_x, T.cast(shared_y, "int32")
	return [shared(training_data), shared(validation_data), shared(test_data)]

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
print("[INFO] downloading MNIST...")
dataset = datasets.fetch_mldata("MNIST Original")

training_data, validation_data, test_data = load_data_shared("data/data1005.pkl.gz")

td , tl = training_data
td = td.get_value();
td = td * 255
td = td.astype('uint16')
tde = td.reshape(-1,20)
tl = tl.eval();
print "tde shape -> ",td.shape
# reshape the MNIST dataset from a flat list of 784-dim vectors, to
# 20 x 20 pixel images, then scale the data to the range [0, 1.0]
# and construct the training and testing splits
print "dataset.data's type -> ",type(dataset.data)," td's type -> ",type(td)
dataset.data=td
dataset.target = tl

data = dataset.data.reshape((dataset.data.shape[0], 20, 20))
print "data shape-> ",data.shape
data = data[:, :, :, np.newaxis]
print data.shape
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data / 255.0, dataset.target.astype("int"), test_size=0.33)

print "shape-> ",(trainData.shape);
print type(testData);
print type(trainLabels);
print type(testLabels);



#print trainData;

# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels
trainLabels = np_utils.to_categorical(trainLabels, 3)
testLabels = np_utils.to_categorical(testLabels, 3)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.02)
model = LeNet.build( width=20, height=20, depth=1, classes=3, weightsPath=args["weights"] if args["load_model"] > 0 else None )
model.compile( loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"] )



'''
for i in range(15):#np.random.choice(np.arange(0, len(testLabels)), size=(50,)):
	# classify the digit
	#probs = model.predict(testData[np.newaxis, i])
	#prediction = probs.argmax(axis=1)
	
	e = np.array( testData[i]*255 ).reshape(-1,20);
	#e = vec2matrix(e,nCol=20)
	image =  (e).astype("uint8")
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, str(np.argmax(testLabels[i])), (5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

	# resize the image from a 20 x 20 image to a 96 x 96 image so we
	# can better see it
	#image = (testData[i] * 255).astype("uint8")
	#image = cv2.merge([image] * 3)
	#image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	#cv2.putText(image, str(prediction[0]), (5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

	# show the image and prediction
	print("[INFO] Actual: {}".format(np.argmax(testLabels[i])))
	cv2.imshow("Digit", image)
	cv2.waitKey(0)
'''




# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args["load_model"] < 0:
	print("[INFO] training...")
	model.fit(trainData, trainLabels, batch_size=82, nb_epoch=16,verbose=1)

	# show the accuracy on the testing set
	print("[INFO] evaluating...")
	(loss, accuracy) = model.evaluate(testData, testLabels,
		batch_size=128, verbose=1)
	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
(loss, accuracy) = model.evaluate(testData, testLabels,batch_size=128, verbose=1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
# check to see if the model should be saved to file
if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)

# randomly select a few testing digits
for i in range(15):#np.random.choice(np.arange(0, len(testLabels)), size=(50,)):
	# classify the digit
	probs = model.predict(testData[np.newaxis, i])
	prediction = probs.argmax(axis=1)
	
	e = np.array( testData[i]*255 ).reshape(-1,20);
	#e = vec2matrix(e,nCol=20)
	image =  (e).astype("uint8")
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, str(prediction[0]), (5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

	# resize the image from a 20 x 20 image to a 96 x 96 image so we
	# can better see it
	#image = (testData[i] * 255).astype("uint8")
	#image = cv2.merge([image] * 3)
	#image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	#cv2.putText(image, str(prediction[0]), (5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

	# show the image and prediction
	print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],np.argmax(testLabels[i])))
	cv2.imshow("Digit", image)
	cv2.waitKey(0)
	

print 'Executed successfully\n'
















