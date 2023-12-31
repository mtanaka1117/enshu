import os.path
import h5py
import re
import random
import numpy as np
from argparse import ArgumentParser
from collections import Counter
from typing import Iterable, Dict, Any


SEQ_LENGTH = 120
STRIDE = 36

CHUNK_SIZE = 5000
TRAIN_FRAC = 0.85
VALID_FRAC = 0.15

LABEL_MAP = {
    'cobblestone': 0,
    'dirt': 1,
    'flexible': 2
}

LINE_REGEX = re.compile(r'[,:]+')


def iterate_dataset(path: str) -> Iterable[Any]:
    with open(path, 'r') as fin:
        is_header = True
        num_features = 0
        num_samples = 0

        for line in fin:
            line = line.strip().lower()

            if line == '@data':
                is_header = False
            elif not is_header:
                tokens = LINE_REGEX.split(line)

                label = LABEL_MAP[tokens[-1]]
                values = tokens[:-1]

                for start_idx in range(0, len(values), STRIDE):
                    features_list = list(map(float, values[start_idx:start_idx+SEQ_LENGTH]))

                    if len(features_list) != SEQ_LENGTH:
                        continue

                    features = np.array(features_list)  # [120]
                    features = np.expand_dims(features, axis=-1)  # [120, 1]

                    yield features, label


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
    args = parser.parse_args()

    # Set the random seed for reproducible results
    random.seed(42)

    # Create the output folder
    # make_dir(args.output_folder)

    train_path = os.path.join(args.input_folder, 'AsphaltPavementType_TRAIN.ts')
    write_dataset(train_path, args.output_folder, series='train')

    test_path = os.path.join(args.input_folder, 'AsphaltPavementType_TEST.ts')
    write_dataset(test_path, args.output_folder, series='test')
