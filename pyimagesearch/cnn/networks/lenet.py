# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Reshape
from keras.layers.convolutional import UpSampling2D
from keras.layers import MaxPooling2D
import numpy as np
from keras.layers import Dropout
from keras import backend as K
from keras.layers.normalization import BatchNormalization

class LeNet:


	@staticmethod
	def build(width, height, depth, classes, weightsPath=None):

		def relu_advanced(x):
			return K.relu(x)
		# initialize the model
		model = Sequential()

		# first set of CONV => RELU => POOL
		model.add(Convolution2D(20, 5, 5, border_mode="same",
			input_shape=( width,height,depth)))
		#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(BatchNormalization())
		model.add(Activation(relu_advanced))
		#model.add(Activation("softplus"))
		

		# second set of CONV => RELU => POOL
		model.add(Convolution2D(20, 10, 10, border_mode="same"))
		#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(BatchNormalization())
		model.add(Activation(relu_advanced))

		model.add(Convolution2D(10, 10, 10, border_mode="same"))
		#model.add(UpSampling2D())
		model.add(BatchNormalization())
		model.add(Activation(relu_advanced))

		model.add(Convolution2D(5, 5, 5, border_mode="same"))
		#model.add(UpSampling2D())
		model.add(BatchNormalization())
		model.add(Activation(relu_advanced))
		#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# set of FC => RELU layers
		model.add(Dropout(0.1))		

		# softmax classifier
		#model.add(Activation("softplus"))
		model.add(Convolution2D(1, 4, 4, border_mode="same"))
		model.add(BatchNormalization())
		model.summary()


		# if a weights path is supplied (inicating that the model was
		# pre-trained), then load the weights
		
		#print model.get_weights()
		current_weights = model.get_weights()
		rand_weights = []
		for i in range(len(current_weights)):
			#print np.shape(current_weights[i]), type(current_weights[i]) 
			rand_weights.append( np.random.random_sample(current_weights[i].shape))

		
		if weightsPath is not None:
			print "Available:",weightsPath
			model.load_weights(weightsPath)
		else:
			model.set_weights(rand_weights)
		
		current_weights = model.get_weights()
		#for i in range(len(current_weights)):
			#print np.shape(current_weights[i]), current_weights[i]
		#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
		return [model,current_weights]
