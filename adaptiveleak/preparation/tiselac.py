import os
import h5py
import re
import random
import numpy as np
from argparse import ArgumentParser
from collections import Counter
from typing import Iterable, Dict, Any


TRAIN_FRAC = 0.85
VALID_FRAC = 0.15
CHUNK_SIZE = 1000
SEQ_LENGTH = 23

LINE_REGEX = re.compile(r"'(.*)',([0-9]+)")
SPLIT_REGEX = re.compile(r'\\n')


def iterate_dataset(path: str) -> Iterable[Any]:
    with open(path, 'r') as fin:
        is_header = True

        for line in fin:
            line = line.strip()

            if is_header:
                is_header = (line != '@data')
            elif len(line) > 0:
                match = LINE_REGEX.match(line)
                features = match.group(1)
                label = int(match.group(2)) - 1

                feature_vec_strings = list(filter(lambda t: len(t) > 0, SPLIT_REGEX.split(features)))
                feature_vectors = np.array([list(map(int, vec.split(','))) for vec in feature_vec_strings])  # [10, 23]

                feature_vectors = np.ascontiguousarray(np.transpose(feature_vectors))  # [23, 10]

                yield feature_vectors, label


def write_dataset(path: str, output_folder: str, series: str):
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

    # Iterate over the dataset
    for index, sample in enumerate(iterate_dataset(path=path)):

        # Select the partition
        if series == 'train':
            if random.random() < TRAIN_FRAC:
                partition = 'train'
            else:
                partition = 'validation'
        else:
            partition = 'test'

        sample_inputs, sample_output = sample

        inputs[partition].append(np.expand_dims(sample_inputs, axis=0))
        output[partition].append(sample_output)

        label_counters[partition][sample_output] += 1

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

        print(partition_inputs.shape)

        with h5py.File(os.path.join(partition_folder, 'data.h5'), 'w') as fout:
            input_dataset = fout.create_dataset('inputs', partition_inputs.shape, dtype='i')
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

    train_path = os.path.join(args.input_folder, '{0}_TRAIN.arff'.format(args.dataset_name))
    write_dataset(train_path, args.output_folder, series='train')

    test_path = os.path.join(args.input_folder, '{0}_TEST.arff'.format(args.dataset_name))
    write_dataset(test_path, args.output_folder, series='test')
