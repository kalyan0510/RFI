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
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))

def dispImg(imarray,sz,dsz):
	e = np.array( imarray ).reshape(-1,sz);
	image =  (e).astype("uint8")
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (dsz, dsz), interpolation=cv2.INTER_LINEAR)
	#cv2.putText(image, ' ' ,(5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
	cv2.imshow("image", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def getArr(model,imarray,sz):
	z = imarray.reshape(sz,sz)
	z = blockshaped(z,20,20)
	print z.shape
	z = z[:,:,:,np.newaxis]
	#z = z.reshape(sz*sz/20,20)
	res = []
	for i in range(z.shape[0]):
		probs = model.predict(z[np.newaxis, i])
		prediction = probs.argmax(axis=1)
		res.append(prediction)
	return res

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


dataset = datasets.fetch_mldata("MNIST Original")

training_data, validation_data, test_data = load_data_shared("data/data1005.pkl.gz")

td , tl = training_data
td = td.get_value();
td = td * 255
td = td.astype('uint16')
tde = td.reshape(-1,20)
tl = tl.eval();

dataset.data=td
dataset.target = tl

data = dataset.data.reshape((dataset.data.shape[0], 20, 20))

data = data[:, :, :, np.newaxis]

(trainData, testData, trainLabels, testLabels) = train_test_split(data / 255.0, dataset.target.astype("int"), test_size=0.33)

trainLabels = np_utils.to_categorical(trainLabels, 2)
testLabels = np_utils.to_categorical(testLabels, 2)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.02)
model = LeNet.build( width=20, height=20, depth=1, classes=2, weightsPath='output/lenet_weights.hdf5')
model.compile( loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"] )

#print model.predict(testData[np.newaxis, i]).argmax(axis=1)

