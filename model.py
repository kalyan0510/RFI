from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
	layer = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
		padding="same")(input_tensor)
	if batchnorm:
	   layer = BatchNormalization()(layer)
	layer = Activation("relu")(layer)
	# second layer
	layer = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
		padding="same")(layer)
	if batchnorm:
	   layer = BatchNormalization()(layer)
	layer = Activation("relu")(layer)
	return layer

def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
	# contracting path
	c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
	p1 = MaxPooling2D((2, 2)) (c1)
	p1 = Dropout(dropout*0.5)(p1)

	c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
	p2 = MaxPooling2D((2, 2)) (c2)
	p2 = Dropout(dropout)(p2)

	c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
	p3 = MaxPooling2D((2, 2)) (c3)
	p3 = Dropout(dropout)(p3)

	c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

	u5 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c4)
	u5 = concatenate([u5, c3])
	u5 = Dropout(dropout)(u5)
	u5 = conv2d_block(u5, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

	u6 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (u5)
	u6 = concatenate([u6, c2])
	u6 = Dropout(dropout)(u6)
	c6 = conv2d_block(u6, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

	u7 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c6)
	u7 = concatenate([u7, c1], axis=3)
	u7 = Dropout(dropout)(u7)
	c7 = conv2d_block(u7, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
	
	outputs = Conv2D(1, (1, 1), activation='sigmoid') (c7)
	model = Model(inputs=[input_img], outputs=[outputs])
	return model