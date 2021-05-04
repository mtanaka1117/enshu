#!/bin/python3

import os.path
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser

from adaptiveleak.policies import AdaptivePolicy, EncodingMode, run_policy, make_policy
from adaptiveleak.utils.constants import SMALL_NUMBER, BIG_NUMBER
from adaptiveleak.utils.encryption import EncryptionMode
from adaptiveleak.utils.loading import load_data
from adaptiveleak.utils.file_utils import iterate_dir, read_json, save_pickle_gz, read_pickle_gz


EPSILON = 1e-3


def execute(policy: AdaptivePolicy, inputs: np.ndarray, batch_size: int, upper: float) -> float:
    seq_length = inputs.shape[1]
    sample_idx = np.arange(inputs.shape[0])

    rand = np.random.RandomState(seed=581)

    observed = BIG_NUMBER
    lower = -upper
    
    current = 0.0
    best_threshold = upper
    best_observed = 0.0

    best_diff = BIG_NUMBER
    batch_size = min(len(sample_idx), batch_size)
    target = policy._target

    while (abs(observed - target) > EPSILON) or (best_observed > target):

        current = (upper + lower) / 2
        
        batch_idx = rand.choice(sample_idx, size=batch_size, replace=False)
        sample_inputs = inputs[batch_idx]

        sample_count = 0
        seq_count = 0

        policy._threshold = current

        # Execute the policy on each sequence
        for seq_idx, sequence in enumerate(sample_inputs):
            policy.reset()
            _, collected_indices = run_policy(policy=policy, sequence=sequence)

            sample_count += len(collected_indices)
            seq_count += 1

        observed = sample_count / (seq_count * seq_length)
        print('Observed Average: {0:.5f}, Current: {1:.5f}'.format(observed, current))

        diff = abs(target - observed)
        if (diff < best_diff) and (observed <= target):
            best_threshold = current
            best_observed = observed
            best_diff = diff

        if (observed < policy._target):
            upper = current
        else:
            lower = current

        if (upper - lower) < SMALL_NUMBER:
            break

    return best_threshold


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--targets', type=float, nargs='+', required=True)
    parser.add_argument('--max-threshold', type=float, default=10.0)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    # Load the data
    inputs, _ = load_data(args.dataset, fold='validation')

    # Load the parameter files
    output_file = os.path.join('saved_models', args.dataset, 'thresholds.pkl.gz')
    threshold_map = read_pickle_gz(output_file) if os.path.exists(output_file) else defaultdict(dict)

    for target in args.targets:
        print('Starting {0}'.format(target))

        policy = make_policy(name=args.policy,
                             target=target,
                             seq_length=inputs.shape[1],
                             num_features=inputs.shape[2],
                             encryption_mode=EncryptionMode.STREAM,
                             encoding='standard',
                             dataset=args.dataset)

        threshold = execute(policy=policy, inputs=inputs, batch_size=args.batch_size, upper=args.max_threshold)
        threshold_map[args.policy][target] = threshold

        print('==========')

    print(threshold_map)

    # Save the results
    save_pickle_gz(threshold_map, output_file)
