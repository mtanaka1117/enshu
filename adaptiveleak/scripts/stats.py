import h5py
import numpy as np


with h5py.File('../datasets/strawberry/all/data.h5', 'r') as fin:
    inputs = fin['inputs'][:]
    output = fin['output'][:].reshape(-1)


print(inputs[1, 0])
print(output.shape)
print(len(np.unique(output)))
print(inputs.shape)
