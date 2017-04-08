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
mini_batch_size = 5
size_x = 28
size_y = 28
num_possible_outcomes = 4
num_pixels = size_x * size_y
#net = Network([FullyConnectedLayer(n_in=num_pixels, n_out=100), SoftmaxLayer(n_in=100, n_out=num_possible_outcomes)], mini_batch_size) 

net = Network(
		[	
			ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(20, 1, 5, 5), poolsize=(2, 2)),
			FullyConnectedLayer(n_in=20*12*12, n_out=100),
			SoftmaxLayer(n_in=100, n_out=num_possible_outcomes)
		],
		mini_batch_size
	     )
net.SGD(training_data, 120, mini_batch_size, 0.1, validation_data, test_data)









