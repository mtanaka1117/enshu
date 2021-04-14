import os
import h5py
import re
import random
import numpy as np
from argparse import ArgumentParser
from collections import Counter
from typing import Iterable, Dict, Any


TRAIN_FRAC = 1.0
VALID_FRAC = 0.15
CHUNK_SIZE = 250

LABEL_MAP = {
    'EPILEPSY': 0,
    'WALKING': 1,
    'RUNNING': 2,
    'SAWING': 3
}


def iterate_dataset(path: str) -> Iterable[Any]:
    with open(path, 'r') as fin:
        is_header = True

        for line in fin:
            if line.strip() == '@data':
                is_header = False
            elif not is_header:
                tokens = line.strip().split(',')
            
                label = LABEL_MAP[tokens[-1]]
                features = list(map(float, tokens[:-1]))

                yield np.array(features).reshape(-1, 1), label


def write_dataset(dim1_path: str, dim2_path: str, dim3_path: str, output_folder: str, series: str):
    # Create the data writers
    if series == 'train':
        inputs = dict(train=[], validation=[])
        output = dict(train=[], validation=[])

        label_counters = {
            'train': Counter(),
            'validation': Counter()
        }
    else:
        inputs = dict(test=[])
        output = dict(test=[])

        label_counters = {
            'test': Counter()
        }

    # Create feature iterators
    dim1_features = iterate_dataset(path=dim1_path)
    dim2_features = iterate_dataset(path=dim2_path)
    dim3_features = iterate_dataset(path=dim3_path)

    # Iterate over the dataset
    for index, (dim1, dim2, dim3) in enumerate(zip(dim1_features, dim2_features, dim3_features)):

        # Select the partition
        if series == 'train':
            if random.random() < TRAIN_FRAC:
                partition = 'train'
            else:
                partition = 'validation'
        else:
            partition = 'test'

        sample_features = np.concatenate([dim1[0], dim2[0], dim3[0]], axis=-1)  # [T, 3]
        sample_label = dim1[1]

        inputs[partition].append(np.expand_dims(sample_features, axis=0))
        output[partition].append(sample_label)

        label_counters[partition][sample_label] += 1

        if (index + 1) % CHUNK_SIZE == 0:
            print('Completed {0} samples.'.format(index + 1), end='\r')
    print()

    print(label_counters)
   
    for fold in inputs.keys():

        if len(inputs[fold]) == 0:
            print('WARNING: Empty fold {0}'.format(fold))
            continue

        partition_inputs = np.vstack(inputs[fold])  # [N, T, D]
        partition_output = np.vstack(output[fold]).reshape(-1)  # [N]

        partition_folder = os.path.join(output_folder, fold)

        if not os.path.exists(partition_folder):
            os.mkdir(partition_folder)

        with h5py.File(os.path.join(partition_folder, 'data.h5'), 'w') as fout:
            input_dataset = fout.create_dataset('inputs', partition_inputs.shape, dtype='f')
            input_dataset.write_direct(partition_inputs)

            output_dataset = fout.create_dataset('output', partition_output.shape, dtype='i')
            output_dataset.write_direct(partition_output)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, required=True)
    args = parser.parse_args()

    # Set the random seed for reproducible results
    random.seed(42)

    # Create the output folder
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    write_dataset(dim1_path=os.path.join(args.input_folder, '{0}Dimension1_TRAIN.arff'.format(args.dataset_name)),
                  dim2_path=os.path.join(args.input_folder, '{0}Dimension2_TRAIN.arff'.format(args.dataset_name)),
                  dim3_path=os.path.join(args.input_folder, '{0}Dimension3_TRAIN.arff'.format(args.dataset_name)),
                  output_folder=args.output_folder,
                  series='train')

    write_dataset(dim1_path=os.path.join(args.input_folder, '{0}Dimension1_TEST.arff'.format(args.dataset_name)),
                  dim2_path=os.path.join(args.input_folder, '{0}Dimension2_TEST.arff'.format(args.dataset_name)),
                  dim3_path=os.path.join(args.input_folder, '{0}Dimension3_TEST.arff'.format(args.dataset_name)),
                  output_folder=args.output_folder,
                  series='test')
