import gzip
import cPickle
import theano
import theano.tensor as T
import numpy as np
import cv2
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

training_data, validation_data, test_data = load_data_shared("data/data1005.pkl.gz")
print type(training_data);
print type(validation_data);
print type(test_data);

t = training_data;
x,y = t
print "type x -> ",type(y),type(x);
x = x.get_value();
y = y.eval();
print x[0];
print len(x);
for i in range(30):
	# classify the digit
	e = np.array( x[i]*255 ).reshape(-1,20);
	#e = vec2matrix(e,nCol=20)
	image =  (e).astype("uint8")
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, str(y[i]), (5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

	# show the image and prediction
	#print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],np.argmax(testLabels[i])))
	cv2.imshow("Digit", image)
	cv2.waitKey(0)
