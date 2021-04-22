"""
Script to compare the differences between standard and encoded policies.
This task is useful for debugging.
"""
from argparse import ArgumentParser
import numpy as np

from adaptiveleak.utils.file_utils import read_json_gz


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--standard', type=str, required=True)
    parser.add_argument('--encoded', type=str, required=True)
    parser.add_argument('--max-num-samples', type=int)
    args = parser.parse_args()

    standard = read_json_gz(args.standard)
    encoded = read_json_gz(args.encoded)

    standard_errors = standard['errors']
    encoded_errors = encoded['errors']

    if args.max_num_samples is not None:
        standard_errors = standard_errors[:args.max_num_samples]
        encoded_errors = encoded_errors[:args.max_num_samples]

    assert len(standard_errors) == len(encoded_errors), 'Misaligned lists ({0} vs {1})'.format(len(standard_errors), len(encoded_errors))

    diff = [e - s for s, e in zip(standard_errors, encoded_errors)]

    max_diff_idx = np.argmax(diff)
    max_diff = diff[max_diff_idx]

    print('Max Diff: {0}'.format(max_diff))
    print('Max Diff Idx: {0}'.format(max_diff_idx))

