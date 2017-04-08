import network
from network import Network

training_data, validation_data, test_data = network.load_data_shared("data/data1005.pkl.gz")
print type(training_data);
print type(validation_data);
print type(test_data);
import numpy as np
import cv2
t = training_data;
x,y = t
print np.asarray(x[0].eval());
print len(np.asarray(x[0].eval()));
for i in range(300):
	# classify the digit
	e = np.array( np.asarray(x[i].eval())*255 ).reshape(-1,28);
	#e = vec2matrix(e,nCol=28)
	image =  (e).astype("uint8")
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, str(i), (5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

	# show the image and prediction
	#print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],np.argmax(testLabels[i])))
	cv2.imshow("Digit", image)
	cv2.waitKey(0)
