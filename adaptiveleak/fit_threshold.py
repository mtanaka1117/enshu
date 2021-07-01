#!/bin/python3

import os.path
import numpy as np
import time
from collections import defaultdict, namedtuple
from argparse import ArgumentParser

from adaptiveleak.server import reconstruct_sequence
from adaptiveleak.policies import run_policy, BudgetWrappedPolicy
from adaptiveleak.energy_systems import convert_rate_to_energy
from adaptiveleak.utils.constants import SMALL_NUMBER, BIG_NUMBER
from adaptiveleak.utils.types import EncodingMode, EncryptionMode
from adaptiveleak.utils.loading import load_data
from adaptiveleak.utils.file_utils import iterate_dir, read_json, save_json_gz, read_json_gz


BatchResult = namedtuple('BatchResult', ['mae', 'did_exhaust'])
VAL_BATCH_SIZE = 2048
MAX_ITER = 150  # Prevents any unexpected infinite looping
THRESHOLD_FACTOR_UPPER = 1.5
THRESHOLD_FACTOR_LOWER = 0.5
TOLERANCE = 1e-5


def execute_on_batch(policy: BudgetWrappedPolicy, batch: np.ndarray, energy_margin: float) -> BatchResult:
    policy.init_for_experiment(num_sequences=batch.shape[0])

    policy._budget -= energy_margin

    # Execute the policy on each sequence
    estimated_list: List[np.ndarray] = []

    for seq_idx, sequence in enumerate(batch):
        policy.reset()
        policy_result = run_policy(policy=policy, sequence=sequence, should_enforce_budget=True)

        # Reconstruct the sequence elements, [T, D]
        reconstructed = reconstruct_sequence(measurements=policy_result.measurements,
                                             collected_indices=policy_result.collected_indices,
                                             seq_length=seq_length)

        estimated_list.append(np.expand_dims(reconstructed, axis=0))

    estimated = np.vstack(estimated_list)  # [B, T, D]

    # Compute the error over the batch
    error = np.average(np.abs(batch - estimated))

    return BatchResult(mae=error,
                       did_exhaust=policy.has_exhausted_budget())


def fit(policy: BudgetWrappedPolicy,
        inputs: np.ndarray,
        batch_size: int,
        lower: float,
        upper: float,
        batches_per_trial: int,
        energy_margin: float,
        should_print: bool) -> float:
    assert batches_per_trial >= 1, 'The # of Batches per Trial must be positive'

    seq_length = inputs.shape[1]
    sample_idx = np.arange(inputs.shape[0])

    rand = np.random.RandomState(seed=581)

    observed = BIG_NUMBER
    
    current = (lower + upper) / 2
    best_threshold = upper
    best_error = BIG_NUMBER

    batch_size = min(len(sample_idx), batch_size)

    # No need to run multiple batches when we are already
    # using the full data-set
    if batch_size == len(sample_idx):
        batches_per_trial = 1

    iter_count = 0
    while (iter_count < MAX_ITER) and (abs(upper - lower) > TOLERANCE):
        # Set the current threshold
        current = (upper + lower) / 2
        
        # Set the threshold for the policy
        policy.set_threshold(threshold=current)

        # Make lists to track results
        did_exhaust_list: List[bool] = []
        error_list: List[float] = []

        # Execute the policy on each batch
        for _ in range(batches_per_trial):
            # Make the batch
            batch_idx = rand.choice(sample_idx, size=batch_size, replace=False)
            batch = inputs[batch_idx]

            batch_result = execute_on_batch(policy=policy, batch=batch, energy_margin=energy_margin)

            error_list.append(batch_result.mae)
            did_exhaust_list.append(batch_result.did_exhaust)

        # Get aggregate metrics
        error = np.average(error_list)
        did_exhaust = any(did_exhaust_list)

        # Track the best error
        if error < best_error:
            best_threshold = current
            best_error = error

        if should_print:
            print('Best Error: {0:.4f}, Best Threshold: {1:.4f}'.format(best_error, best_threshold), end='\r')

        # Get the search direction based on the budget use
        budget = policy.budget
        consumed_energy = policy.consumed_energy

        if did_exhaust:
            lower = current
        else:
            upper = current

        # Ensure that lower <= upper
        temp = min(lower, upper)
        upper = max(lower, upper)
        lower = temp
        iter_count += 1

    if should_print:
        print()

    return best_threshold


def validate_thresholds(policy: BudgetWrappedPolicy, inputs: np.ndarray, threshold: float, energy_margin: float) -> BatchResult:
    """
    Validates the policy and thresholds on a set of held-out inputs.
    """
    # Set the threshold
    policy._threshold = threshold

    # Run the policy on the given batch
    val_result = execute_on_batch(policy=policy, batch=inputs, energy_margin=energy_margin)

    return val_result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--collection-rates', type=float, nargs='+', required=True)
    parser.add_argument('--encoding', type=str, required=True, choices=['standard' , 'group'])
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--batches-per-trial', type=int, default=3)
    parser.add_argument('--should-print', action='store_true')
    args = parser.parse_args()

    # Load the data. We fit thresholds on the "validation" set and
    # validation on the "training" set. Fitting the thresholds on the training
    # set can cause overfitting because some policies (e.g. Skip RNNs) fit their policy
    # to the training set.
    inputs, _ = load_data(args.dataset, fold='validation')
    val_inputs, _ = load_data(args.dataset, fold='train')

    # Unpack the data dimensions
    num_seq, seq_length, num_features = inputs.shape

    # Get the maximum threshold value based on the data
    max_threshold = np.max(np.abs(inputs)) * 1.01

    # Load the parameter files
    output_file = os.path.join('saved_models', args.dataset, 'thresholds.json.gz')
    threshold_map = read_json_gz(output_file) if os.path.exists(output_file) else dict()

    policy_name = args.policy
    encoding = args.encoding

    if policy_name not in threshold_map:
        threshold_map[policy_name] = dict()

    if encoding not in threshold_map[policy_name]:
        threshold_map[policy_name][encoding] = dict()

    # Create parameters for policy validation data splitting
    val_indices = np.arange(val_inputs.shape[0])
    rand = np.random.RandomState(seed=3485)

    for collection_rate in args.collection_rates:

        # Set the lower threshold based on the model type
        if policy_name == 'skip_rnn':
            lower = 0.0
            upper = 1.0
        else:
            lower = -1 * max_threshold
            upper = max_threshold

        # Always log the progress for intermediate tracking
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

        final_threshold = None
        energy_margin = 1
        did_exhaust = True

        while (did_exhaust):
            # Fit the policy using the given rate and energy margin
            threshold = fit(policy=policy,
                            inputs=inputs,
                            batch_size=args.batch_size,
                            lower=lower,
                            upper=upper,
                            batches_per_trial=args.batches_per_trial,
                            energy_margin=energy_margin,
                            should_print=args.should_print)

            # Make the validation batch
            if len(val_indices) > VAL_BATCH_SIZE:
                val_batch_idx = rand.choice(val_indices, size=VAL_BATCH_SIZE, replace=True)
            else:
                val_batch_idx = val_indices

            val_margin = energy_margin * 0.5
            val_batch = val_inputs[val_batch_idx]
            val_result = validate_thresholds(policy=policy,
                                             threshold=threshold,
                                             inputs=val_batch,
                                             energy_margin=val_margin)

            did_exhaust = val_result.did_exhaust
            final_threshold = threshold

            # Reset the bounds to speed up the next iteration
            upper = threshold * THRESHOLD_FACTOR_UPPER
            lower = threshold * THRESHOLD_FACTOR_LOWER

            energy_margin += 1

        threshold_map[policy_name][encoding][str(collection_rate)] = final_threshold

        if args.should_print:
            print('==========')

    print(threshold_map)

    # Save the results
    save_json_gz(threshold_map, output_file)
