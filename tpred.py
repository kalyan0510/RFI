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
import cPickle as pickle
import gzip
import sys
def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))


# Print iterations progress
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100, fill = '\xe2'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(barLength * iteration // total)
    bar = fill * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def dispImg(imarray,sz,dsz,loc):
	e = np.array( imarray ).reshape(-1,sz);
	image =  (e).astype("uint8")
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (dsz, dsz), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, loc,(5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
	cv2.imshow("image", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def getArr(model,imarray,sz):
	z = imarray.reshape(sz,sz)
	z = blockshaped(z,20,20)
	#print z.shape
	z = z[:,:,:,np.newaxis]
	#z = z.reshape(sz*sz/400,400)
	res = []
	x = 0
	for i in range(z.shape[0]):
		probs = model.predict(z[np.newaxis, i])
		prediction = probs.argmax(axis=1)
		res.append(prediction[0])
	x = np.array(res).reshape(sz/20,sz/20).sum(axis=1).argmax()
	#print np.array(res).reshape(sz/20,sz/20);
	if(np.array(res).reshape(sz/20,sz/20).sum(axis=1).max() == 0):
		print 0;
		return 0
	print x*20 + 10
	return x*20 + 10

def load_data_shared(filename="data/mnist.pkl.gz"):
	f = gzip.open(filename, 'rb')
	training_data = pickle.load(f)
	f.close()
	return training_data

print 'loading data(750MB)'
training_data = load_data_shared("data/data1005.pkl.gz")

td , tl , ts, tw = training_data
#td = td.get_value();
td = td * 255
td = td.astype('uint16')

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.02)
model = LeNet.build( width=20, height=20, depth=1, classes=2)
model.compile( loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"] ,weightsPath='output/lenet_weights.hdf5')

#print model.predict(testData[np.newaxis, i]).argmax(axis=1)
td = td.astype('float64')/255.0
# 	getArr(model,td[12],600)
pl = []
for i in range(20):
	pl.append(getArr(model,td[i],600))
	printProgress(i+1, 30, prefix = 'Progress:', suffix = 'Complete', barLength = 70)

#er = np.absolute(np.array(ts)-np.array(pl))/1.0
#er = er/600.0
#sprint '\nAvg Accuracy in extracting location from 200 images of (600x600) is :',100-(np.array(er).sum()/2.0),'%'

for i in range(15):
	dispImg(td[i]*255.0,600,600,str(pl[i]) if pl[i]!=0 else 'no rfi')

print 'predicted values \n',pl
print 'actual values    \n',ts


f = gzip.open("data/pred_res.pkl.gz", "w")
pickle.dump((pl), f)
f.close()



