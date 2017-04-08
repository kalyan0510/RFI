import cPickle
import gzip

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

#### Load the MNIST data
def load_data_shared(filename="data/mnist.pkl.gz"):
	f = gzip.open(filename, 'rb')
	training_data, validation_data, test_data = cPickle.load(f)
	f.close()
	return [training_data, validation_data,test_data]


#### Miscellanea
def size(data):
	"Return the size of the dataset `data`."
	return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
	srng = shared_randomstreams.RandomStreams(
		np.random.RandomState(0).randint(999999))
	mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
	return layer*T.cast(mask, theano.config.floatX)
