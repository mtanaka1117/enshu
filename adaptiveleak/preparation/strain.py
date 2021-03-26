import numpy as np
import h5py


def read_samples(path: str):

    input_list: List[np.ndarray] = []
    output_list: List[int] = []

    with open(path, 'r') as fin:
        for line in fin:
            values = list(map(float, line.split()))

            label = int(values[0])
            features = values[1:]

            input_list.append(np.expand_dims(features, axis=0))
            output_list.append(label)

    return np.vstack(input_list), np.vstack(output_list)

fold = 'test'
path = '/home/tejask/Downloads/MoteStrain/MoteStrain_{0}.txt'.format(fold.upper())
output_path = '../datasets/strain/{0}/data.h5'.format(fold)

inputs, outputs = read_samples(path)

with h5py.File(output_path, 'w') as fout:
    input_dataset = fout.create_dataset('inputs', inputs.shape,  dtype='f')
    input_dataset.write_direct(inputs)

    output_dataset = fout.create_dataset('output', outputs.shape, dtype='i')
    output_dataset.write_direct(outputs)
