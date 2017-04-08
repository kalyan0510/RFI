import network
from network import Network
from network import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
import cPickle
import gzip

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T


#training_data, validation_data, test_data = network.load_data_shared()
training_data, validation_data, test_data = network.load_data_shared("data/data1005.pkl.gz")
print type(training_data);
print type(validation_data);
print type(test_data);
mini_batch_size = 5
size_x = 28
size_y = 28
num_possible_outcomes = 4
num_pixels = size_x * size_y
net = Network(
		[
			ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(20, 1, 5, 5), poolsize=(2, 2)),
			ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), filter_shape=(40, 20, 5, 5), poolsize=(2, 2)),
			FullyConnectedLayer(n_in=40*4*4, n_out=100),
			SoftmaxLayer(n_in=100, n_out=10)
		],
		mini_batch_size
	     )
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)










