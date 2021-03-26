import h5py
import numpy as np


with h5py.File('datasets/emg/test/data.h5', 'r') as fin:
    inputs = fin['inputs'][:]
    output = fin['output'][:].reshape(-1)


print(output.shape)
print(len(np.unique(output)))
print(inputs.shape)
