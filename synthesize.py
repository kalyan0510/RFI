from __future__ import print_function

import sys, random, os, pickle, gzip

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def makePixel(binary, errorprob):
	return (1 if random.randint(1,100) > errorprob*100 else 0) if binary else (random.random()/5+0.8 if random.randint(1,100) > errorprob*100 else random.random()*0.7 )
	

def rfiCurtain(img_name, op_path, size_x, size_y, binary):
	mlenf = 3
	rfi_axis = 'x' if random.random()>0.5 else 'y'
	if rfi_axis == 'x':
		rfi_start = random.randint(1,size_x)
		rfi_width = int(size_x/(10*random.randint(1,size_x)))
		rfi_init =  random.randint(1,size_y-size_y/mlenf)
		rfi_stop = random.randint(rfi_init+size_y/mlenf,size_y)
	else:
		rfi_start = random.randint(1,size_y)
		rfi_width = int(size_y/(10*random.randint(1,size_y)))
		rfi_init =  random.randint(1,size_x-size_x/mlenf)
		rfi_stop = random.randint(rfi_init+size_x/mlenf,size_x)
	rfi_width = random.randint(2,5)	
	img = np.array([0.0 for j in range(size_x*size_y)])
	opg = np.array([0.0 for j in range(size_x*size_y)])
	for i in range(size_y):
		for j in range(size_x):
			intensity = makePixel(binary, 0.5)
			if ((rfi_axis=='y') and (j>=rfi_start and j<=rfi_start+rfi_width)) and (i>=rfi_init and i<=rfi_stop) or ((rfi_axis=='x') and (i>=rfi_start and i<=rfi_start+rfi_width) and (j>=rfi_init and j<=rfi_stop)):
				bias = 0.1 if ((rfi_axis=='y') and (j<=rfi_start+int(rfi_width*0.1) or j>=rfi_start+int(rfi_width*0.1))) or ((rfi_axis=='x') and (i<=rfi_start+int(rfi_width*0.1) and i>=rfi_start+int(rfi_width*0.9))) else 0.05
				intensity = makePixel(binary, bias)
				opg[i*size_x+j] = intensity			
			img[i*size_x+j] = intensity
	plotImg(img, op_path+rfi_axis+'_'+img_name+'.png')
	plotImg(opg, op_path+rfi_axis+'_'+img_name+'_R.png')
	#print(img)
	return (img, opg)

def rotate(imgsrc, a):
	img = [[0 for j in range(size_x)] for i in range(size_y)]
	for i in range(size_y):
		for j in range(size_x):
			img[i][j] = imgsrc[i*size_x+j]
	#H = np.array(img)
	rotated  = np.rot90(img,a)
	flatten = np.array([0.0 for j in range(size_x*size_y)])
	for i in range(size_y):
		for j in range(size_x):
			flatten[i*size_x+j] = rotated[i][j]
	return flatten

def pulsarCurtain(img_name, op_path, size_x, size_y, binary):
	rfi_start_x = random.randint(1,size_x-20)
	rfi_end_x =  random.randint(20+rfi_start_x,size_x)
	rfi_start_y = random.randint(1,size_y-20)
	rfi_end_y = random.randint(20+rfi_start_y,size_y)
	rfi_width = random.randint(3,5)
	img = np.array([0.0 for j in range(size_x*size_y)])
	opg = np.array([0.0 for j in range(size_x*size_y)])
	c_1 = (((rfi_start_x+rfi_end_x)/2.0)*((rfi_start_y+rfi_end_y)/2.0))
	c_2 = (((rfi_start_x+rfi_end_x)/2.0+rfi_width*0.1)*((rfi_start_y+rfi_end_y)/2.0+rfi_width*0.1))
	c_3 = (((rfi_start_x+rfi_end_x)/2.0+rfi_width*0.9)*((rfi_start_y+rfi_end_y)/2.0+rfi_width*0.9))
	c_4 = (((rfi_start_x+rfi_end_x)/2.0+rfi_width)*((rfi_start_y+rfi_end_y)/2.0+rfi_width))
	for i in range(size_y):
		for j in range(size_x):
			intensity = makePixel(binary, 0.5)
			if ((i>=rfi_start_y and i<=rfi_end_y and j>=rfi_start_x and j<=rfi_end_x) and (i*j>c_1 and i*j<c_4)):
				coin = random.random()
				bias = 0.05 if ((i*j>c_1 and i*j<c_2) or (i*j>c_3 and i*j<c_4)) else 0.1
				intensity = makePixel(binary, bias)
				opg[i*size_x+j] = 0			
			img[i*size_x+j] = intensity
	rot = 3
	#print rot
	img = rotate(img,rot)
	opg = rotate(opg,rot)
	plotImg(img, op_path+'P_'+img_name+'.png')
	plotImg(opg, op_path+'P_'+img_name+'_R.png')
	return (img, opg)

def rfiAndPulsar(img_name, op_path, size_x, size_y, binary):
	(pulsar, pulsarFil) = pulsarCurtain(img_name, op_path, size_x, size_y, binary)
	(rfi, rfiFil) = rfiCurtain(img_name, op_path, size_x, size_y, binary)
	im = np.maximum(pulsar, rfiFil)
	plotImg(im, op_path+'Both_'+img_name+'.png')
	plotImg(rfiFil, op_path+'Both_'+img_name+'_R.png')
	return (im, rfiFil)


def plotImg(imgsrc, imgName=''):
	if 1==1:
		return
	img = [[0 for j in range(size_x)] for i in range(size_y)]
	for i in range(size_y):
		for j in range(size_x):
			img[i][j] = imgsrc[i*size_x+j]
	H = np.matrix(img)
	dpi = 100
	figsize = size_x/float(dpi), size_y/float(dpi)
	fig = plt.figure(figsize=figsize)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	plt.imshow(H, interpolation='none', aspect='auto')
	plt.savefig(imgName)

																																																																																


def generateRandomData(img_name, op_path, size_x, size_y, binary):
	if(random.random()>0.5):
		return rfiCurtain(img_name, op_path, size_x, size_y, binary)
	else:
		return pulsarCurtain(img_name, op_path, size_x, size_y, binary)

def generateNormalData(img_name, op_path, size_x, size_y, binary):
	i = 0
	j = 0
	img = [np.float32(0.0) for i in range(size_x*size_y)]
	for i in range(size_y*size_x):
		intensity = makePixel(binary)		
		img[i] = intensity
	plotImg(img, op_path+'N_'+img_name+'.png')
	plotImg([0 for j in range(size_x*size_y)],op_path+'N_'+img_name+'_R.png' )
	return (img, [0 for j in range(size_x*size_y)])

def compressData(samples):
	training_data_x = []
	validation_data_x = []
	test_data_x = []
	training_data_y = []
	validation_data_y = []
	test_data_y = []
	j = 0 # counter
	num_samples = len(samples)
	random.shuffle(samples)
	for data in samples:
		training_data_x.append(data[0])
		training_data_y.append(data[1])
		j+=1
	training_data = [np.array(training_data_x), np.array(training_data_y)]
	
	print("Saving data. This may take a few minutes.")
	f = gzip.open("data/kalyan.pkl.gz", "w")
	pickle.dump((training_data), f)
	f.close()

if __name__ == "__main__":
	if len(sys.argv)!=6:
		print ("Usage: synthesize.py op_path size_x size_y binary num_images")
		exit(-1)

	op_path = sys.argv[1]+"/"
	size_x = int(sys.argv[2])
	size_y = int(sys.argv[3])	
	binary = int(sys.argv[4])
	num_images = int(sys.argv[5])
	
	samples = []
	for imgcnt in range(num_images):
		print("#"+str(imgcnt+1))
		img_name = "img_"+str(imgcnt)
		samples.append(pulsarCurtain(img_name, op_path, size_x, size_y, binary))
		samples.append(rfiCurtain(img_name, op_path, size_x, size_y, binary))
		samples.append(rfiAndPulsar(img_name, op_path, size_x, size_y, binary))
		#samples.append(generateNormalData(img_name, op_path, size_x, size_y, binary))

	print("Data being compressed.")
	compressData(samples)