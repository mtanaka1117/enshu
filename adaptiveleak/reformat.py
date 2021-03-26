import h5py
import os.path
import numpy as np
from typing import Iterable, Tuple, List

from utils.file_utils import make_dir, read_jsonl_gz, iterate_dir


def get_data(input_folder: str) -> Iterable[Tuple[List[float], int]]:
    for path in iterate_dir(input_folder, pattern=r'.*jsonl.gz'):
        for element in read_jsonl_gz(path):
            inputs = element['inputs']
            output = element['output']
            yield inputs, output


def reformat(input_folder: str, output_folder: str):
    make_dir(output_folder)

#    with h5py.File(os.path.join(input_folder, 'data.h5'), 'r') as fin:
#        inputs = fin['inputs'][:].reshape(-1, SEQ_LENGTH, NUM_FEATURES)
#        output = fin['output'][:]

    input_list: List[np.ndarray] = []
    output_list: List[int] = []
    for sample_inputs, sample_output in get_data(input_folder):
        input_list.append(np.expand_dims(sample_inputs, axis=0))
        output_list.append(sample_output)

    inputs = np.vstack(input_list)
    output = np.vstack(output_list).reshape(-1)

    with h5py.File(os.path.join(output_folder, 'data.h5'), 'w') as fout:
        input_dataset = fout.create_dataset('inputs', inputs.shape, dtype='f')
        input_dataset.write_direct(inputs)

        output_dataset = fout.create_dataset('output', output.shape, dtype='i')
        output_dataset.write_direct(output)


# Reformat all folds
input_base = '/home/tejask/Documents/budget-rnn/src/datasets/strawberry/folds_235'
output_base = 'datasets/strawberry'

make_dir(output_base)

reformat(input_folder=os.path.join(input_base, 'train'),
         output_folder=os.path.join(output_base, 'train'))

reformat(input_folder=os.path.join(input_base, 'valid'),
         output_folder=os.path.join(output_base, 'validation'))

reformat(input_folder=os.path.join(input_base, 'test'),
         output_folder=os.path.join(output_base, 'test'))
