#!/bin/python3

import os.path
import numpy as np
import time
from collections import defaultdict, namedtuple
from argparse import ArgumentParser
from typing import List

from adaptiveleak.server import reconstruct_sequence
from adaptiveleak.policies import run_policy, BudgetWrappedPolicy
from adaptiveleak.utils.constants import SMALL_NUMBER, BIG_NUMBER
from adaptiveleak.utils.loading import load_data
from adaptiveleak.utils.file_utils import iterate_dir, read_json, save_json_gz, read_json_gz


BatchResult = namedtuple('BatchResult', ['mae', 'did_exhaust'])
VAL_BATCH_SIZE = 512
MAX_ITER = 100  # Prevents any unexpected infinite looping

MAX_MARGIN_FACTOR = 0.03  # Limit the padding to 3% of the overall budget (~300 mJ)
VAL_MARGIN_FACTOR = 0.9
MARGIN_FACTOR = 0.001  # Go in increments of 0.1% of the total budget (~10mJ)

THRESHOLD_FACTOR_UPPER = 1.75  # Bias toward reducing the energy rate every time we increase the margin
THRESHOLD_FACTOR_LOWER = 0.5
TOLERANCE = 1e-4


def execute_on_batch(policy: BudgetWrappedPolicy, batch: np.ndarray, energy_margin: float) -> BatchResult:
    policy.init_for_experiment(num_sequences=batch.shape[0])

    # Reduce the budget by the given margin factor
    margin = policy._budget * energy_margin
    policy._budget -= margin

    # Execute the policy on each sequence
    estimated_list: List[np.ndarray] = []

    num_collected = 0
    total = 0

    for seq_idx, sequence in enumerate(batch):
        policy.reset()
        policy_result = run_policy(policy=policy, sequence=sequence, should_enforce_budget=True)

        # Reconstruct the sequence elements, [T, D]
        reconstructed = reconstruct_sequence(measurements=policy_result.measurements,
                                             collected_indices=policy_result.collected_indices,
                                             seq_length=seq_length)

        estimated_list.append(np.expand_dims(reconstructed, axis=0))

        num_collected += policy_result.num_collected
        total += len(sequence)

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
        if (error < best_error) and (not did_exhaust):
            best_threshold = current
            best_error = error

        if should_print:
            print('Best Error: {0:.7f}, Best Threshold: {1:.7f}'.format(best_error, best_threshold), end='\r')

        # Get the search direction based on the budget use
        budget = policy.budget
        consumed_energy = policy.consumed_energy

        if did_exhaust:
            lower = current  # Reduce the energy consumption
        else:
            upper = current  # Increase the energy consumption

        # Ensure that lower <= upper
        temp = min(lower, upper)
        upper = max(lower, upper)
        lower = temp
        iter_count += 1

    if should_print:
        print()

    return best_threshold


def validate_thresholds(policy: BudgetWrappedPolicy,
                        inputs: np.ndarray,
                        threshold: float,
                        energy_margin: float,
                        num_batches: int,
                        rand: np.random.RandomState) -> List[BatchResult]:
    """
    Validates the policy and thresholds on a set of held-out inputs.
    """
    # Set the threshold
    policy._threshold = threshold

    # Create the sample indices for batch creation
    sample_idx = np.arange(len(inputs))

    # Reduce the energy margin to account for some variance
    energy_margin *= VAL_MARGIN_FACTOR

    batch_size = VAL_BATCH_SIZE

    if len(inputs) <= VAL_BATCH_SIZE:
        batch_size = len(inputs)
        num_batches = 1

    results: List[BatchResult] = []
    for _ in range(num_batches):
        # Make the validation batch
        batch_idx = rand.choice(sample_idx, size=batch_size, replace=False)
        batch = inputs[batch_idx]

        # Run the policy on the given batch
        val_result = execute_on_batch(policy=policy, batch=batch, energy_margin=energy_margin)

        results.append(val_result)

    return results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True, choices=['adaptive_heuristic', 'adaptive_deviation'])
    parser.add_argument('--collection-rates', type=float, nargs='+', required=True)
    parser.add_argument('--collect', type=str, required=True, choices=['tiny', 'low', 'med', 'high'])
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
    max_threshold = np.max(np.sum(np.abs(inputs), axis=-1)) + 1000.0

    # Load the parameter files
    output_file = os.path.join('saved_models', args.dataset, 'thresholds.json.gz')
    threshold_map = read_json_gz(output_file) if os.path.exists(output_file) else dict()

    policy_name = args.policy
    collect_mode = args.collect

    if policy_name not in threshold_map:
        threshold_map[policy_name] = dict()

    if collect_mode not in threshold_map[policy_name]:
        threshold_map[policy_name][collect_mode] = dict()

    # Create parameters for policy validation data splitting
    val_indices = np.arange(val_inputs.shape[0])
    rand = np.random.RandomState(seed=3485)

    for collection_rate in args.collection_rates:

        # Set the lower threshold based on the model type
        lower = -1 * max_threshold
        upper = max_threshold

        if args.should_print:
            print('Starting {0}'.format(collection_rate))

        # Create the policy for which to fit thresholds
        policy = BudgetWrappedPolicy(name=policy_name,
                                     collection_rate=collection_rate,
                                     seq_length=seq_length,
                                     num_features=num_features,
                                     encryption_mode='stream',
                                     collect_mode=collect_mode,
                                     encoding='standard',
                                     dataset=args.dataset,
                                     should_compress=False)

        final_threshold = None
        energy_margin = MARGIN_FACTOR
        did_exhaust = True

        while (did_exhaust) and (energy_margin < MAX_MARGIN_FACTOR):
            # Fit the policy using the given rate and energy margin
            threshold = fit(policy=policy,
                            inputs=inputs,
                            batch_size=args.batch_size,
                            lower=lower,
                            upper=upper,
                            batches_per_trial=args.batches_per_trial,
                            energy_margin=energy_margin,
                            should_print=args.should_print)

            # Run on the validation set
            val_results = validate_thresholds(policy=policy,
                                              threshold=threshold,
                                              inputs=val_inputs,
                                              energy_margin=energy_margin,
                                              num_batches=args.batches_per_trial,
                                              rand=rand)

            did_exhaust = any(r.did_exhaust for r in val_results)
            final_threshold = threshold

            # Reset the bounds to speed up the next iteration
            upper = threshold * THRESHOLD_FACTOR_UPPER
            lower = threshold * THRESHOLD_FACTOR_LOWER

            energy_margin += MARGIN_FACTOR

        threshold_map[policy_name][collect_mode][str(round(collection_rate, 2))] = final_threshold

        if args.should_print:
            print('==========')

    # Save the results
    save_json_gz(threshold_map, output_file)
