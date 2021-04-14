import h5py
import os.path
import numpy as np
from argparse import ArgumentParser
from typing import Tuple

from adaptiveleak.utils.file_utils import make_dir


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, 'r') as fin:
        inputs = fin['inputs'][:]
        labels = fin['output'][:]

    return inputs, labels


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    dataset_folder = os.path.join('..', 'datasets', args.dataset)

    # Load the data folds
    train_inputs, train_labels = load_dataset(path=os.path.join(dataset_folder, 'train', 'data.h5'))
    test_inputs, test_labels = load_dataset(path=os.path.join(dataset_folder, 'test', 'data.h5'))

    # Merge the folds
    merged_inputs = np.concatenate([train_inputs, test_inputs], axis=0)
    merged_labels = np.concatenate([train_labels, test_labels], axis=0)

    # Write the output
    make_dir(os.path.join(dataset_folder, 'all'))

    with h5py.File(os.path.join(dataset_folder, 'all', 'data.h5'), 'w') as fout:
        input_dataset = fout.create_dataset('inputs', shape=merged_inputs.shape, dtype='f')
        input_dataset.write_direct(merged_inputs)

        output_dataset = fout.create_dataset('output', shape=merged_labels.shape, dtype='i')
        output_dataset.write_direct(merged_labels)
