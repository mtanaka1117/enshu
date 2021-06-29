import h5py
import numpy as np
import os.path

dataset_name = 'strawberry'
fold = 'test'

with h5py.File(os.path.join('..', 'datasets', dataset_name, fold, 'data.h5'), 'r') as fin:
    inputs = fin['inputs'][:]
    output = fin['output'][:].reshape(-1)


print(len(np.unique(output)))
print(inputs.shape)
print(inputs[0][0:10])
