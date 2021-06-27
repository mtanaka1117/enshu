#!/bin/python3

import os.path
import numpy as np
import time
from collections import defaultdict
from argparse import ArgumentParser

from adaptiveleak.server import reconstruct_sequence
from adaptiveleak.policies import run_policy, BudgetWrappedPolicy
from adaptiveleak.energy_systems import convert_rate_to_energy
from adaptiveleak.utils.constants import SMALL_NUMBER, BIG_NUMBER
from adaptiveleak.utils.types import EncodingMode, EncryptionMode
from adaptiveleak.utils.loading import load_data
from adaptiveleak.utils.file_utils import iterate_dir, read_json, save_json_gz, read_json_gz


MARGIN = 2


def execute(policy: BudgetWrappedPolicy,
            inputs: np.ndarray,
            batch_size: int,
            lower: float,
            upper: float,
            should_print: bool) -> float:
    seq_length = inputs.shape[1]
    sample_idx = np.arange(inputs.shape[0])

    rand = np.random.RandomState(seed=581)

    observed = BIG_NUMBER
    
    current = (lower + upper) / 2
    best_threshold = upper
    best_error = BIG_NUMBER

    batch_size = min(len(sample_idx), batch_size)

    while ((upper > lower) and ((upper - lower) > SMALL_NUMBER)):
        # Set the current threshold
        current = (upper + lower) / 2
        
        # Make the batch
        batch_idx = rand.choice(sample_idx, size=batch_size, replace=False)
        sample_inputs = inputs[batch_idx]

        # Set the threshold for the policy
        policy.set_threshold(threshold=current)

        policy.init_for_experiment(num_sequences=batch_size)

        policy._budget -= MARGIN

        # Execute the policy on each sequence
        estimated_list: List[np.ndarray] = []

        for seq_idx, sequence in enumerate(sample_inputs):
            policy.reset()
            policy_result = run_policy(policy=policy, sequence=sequence)

            # Reconstruct the sequence elements, [T, D]
            reconstructed = reconstruct_sequence(measurements=policy_result.measurements,
                                                 collected_indices=policy_result.collected_indices,
                                                 seq_length=seq_length)

            estimated_list.append(np.expand_dims(reconstructed, axis=0))

        estimated = np.vstack(estimated_list)  # [B, T, D]

        # Compute the error over the batch
        error = np.average(np.abs(sample_inputs - estimated))

        if error < best_error:
            best_threshold = current
            best_error = error

        if should_print:
            print('Best Error: {0:.4f}, Best Threshold: {1:.4f}'.format(best_error, best_threshold), end='\r')

        # Get the search direction based on the budget use
        budget = policy.budget
        consumed_energy = policy.consumed_energy

        if consumed_energy > budget:
            lower = current
        else:
            upper = current

    if should_print:
        print()

    return best_threshold


   # while (abs(observed - target_energy) > EPSILON):

   #     current = (upper + lower) / 2
   #     
   #     batch_idx = rand.choice(sample_idx, size=batch_size, replace=False)
   #     sample_inputs = inputs[batch_idx]

   #     sample_count = 0
   #     seq_count = 0

   #     policy._threshold = current

   #     # Execute the policy on each sequence
   #     for seq_idx, sequence in enumerate(sample_inputs):
   #         policy.reset()
   #         policy_result = run_policy(policy=policy, sequence=sequence)

   #         energy_sum += policy_result.energy
   #         seq_count += 1

   #     observed = energy_sum / seq_count
   #     print('Observed Average: {0:.5f}, Current: {1:.5f}'.format(observed, current))

   #     diff = abs(target - observed)
   #     if (diff < best_diff) and (observed <= target):
   #         best_threshold = current
   #         best_observed = observed
   #         best_diff = diff

   #     if (observed < policy._target):
   #         upper = current
   #     else:
   #         lower = current

   #     if (upper - lower) < SMALL_NUMBER:
   #         break

   # return best_threshold


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--collection-rates', type=float, nargs='+', required=True)
    parser.add_argument('--encoding', type=str, required=True, choices=['standard' , 'group'])
    parser.add_argument('--max-threshold', type=float, default=10.0)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--should-print', action='store_true')
    args = parser.parse_args()

    # Load the data
    inputs, _ = load_data(args.dataset, fold='validation')

    # Unpack the data dimensions
    num_seq, seq_length, num_features = inputs.shape

    # Load the parameter files
    output_file = os.path.join('saved_models', args.dataset, 'thresholds.json.gz')
    threshold_map = read_json_gz(output_file) if os.path.exists(output_file) else dict()

    policy_name = args.policy
    encoding = args.encoding

    if policy_name not in threshold_map:
        threshold_map[policy_name] = dict()

    if encoding not in threshold_map[policy_name]:
        threshold_map[policy_name][encoding] = dict()

    # Set the lower threshold based on the model type
    if policy_name == 'skip_rnn':
        lower = 0.0
        upper = 1.0
    else:
        lower = -1 * args.max_threshold
        upper = args.max_threshold

    for collection_rate in args.collection_rates:
        if args.should_print:
            print('Starting {0}'.format(collection_rate))

        # Create the policy for which to fit thresholds
        policy = BudgetWrappedPolicy(name=policy_name,
                                     collection_rate=collection_rate,
                                     seq_length=seq_length,
                                     num_features=num_features,
                                     encryption_mode=EncryptionMode.STREAM,
                                     encoding=encoding,
                                     dataset=args.dataset,
                                     should_compress=False)

        threshold = execute(policy=policy,
                            inputs=inputs,
                            batch_size=args.batch_size,
                            lower=lower,
                            upper=upper,
                            should_print=args.should_print)

        threshold_map[policy_name][encoding][str(collection_rate)] = threshold

        if args.should_print:
            print('==========')

    print(threshold_map)

    # Save the results
    save_json_gz(threshold_map, output_file)
