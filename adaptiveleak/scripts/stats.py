import h5py
import numpy as np
import os.path

dataset_name = 'tiselac'
fold = 'test'

with h5py.File(os.path.join('..', 'datasets', dataset_name, fold, 'data.h5'), 'r') as fin:
    inputs = fin['inputs'][:]
    output = fin['output'][:].reshape(-1)


data_range = np.max(inputs) - np.min(inputs)

print('# Labels: {0}'.format(len(np.unique(output))))
print('Input Shape: {0}'.format(inputs.shape))
print('Range: {0}'.format(data_range))
print(inputs[0][0:10])

