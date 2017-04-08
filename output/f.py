

from __future__ import print_function
import numpy as np
import h5py
 
with h5py.File('lenet_weights.hdf5','r') as hf:
    print('List of arrays in this file: \n', hf.keys())
    data = hf.get('dataset_1')
    np_data = np.array(data)
    print('Shape of the array dataset_1:\n', np_data.shape)


