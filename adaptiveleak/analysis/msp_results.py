import numpy as np
import os.path
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
from argparse import ArgumentParser
from collections import namedtuple
from typing import Dict, List, Set, Any

from adaptiveleak.analysis.plot_utils import PLOT_STYLE, LINE_WIDTH, to_label
from adaptiveleak.policies import BudgetWrappedPolicy
from adaptiveleak.utils.constants import SMALL_NUMBER
from adaptiveleak.utils.file_utils import read_json_gz, iterate_dir, read_json
from adaptiveleak.utils.loading import load_data


Summary = namedtuple('Summary', ['energy', 'comm', 'mae', 'num_collected', 'num_bytes'])
PolicyResult = namedtuple('PolicyResult', ['energy', 'energy_std', 'error', 'p_value', 't_stat'])


def get_overlapping_indices(policy_results: Dict[str, Dict[int, Summary]], rate: int) -> Set[int]:
    indices: Set[int] = set(range(2000))  # Large enough for all tasks

    for policy_result in policy_results.values():
        summary = policy_result[rate]

        num_collected = summary.num_collected
        max_collected = len(summary.energy)

        if num_collected < max_collected:
            indices_to_keep = np.argsort(summary.mae)[:num_collected]
        else:
            indices_to_keep = list(range(max_collected))

        index_set = set(indices_to_keep)
        indices = indices.intersection(index_set)

    return indices


def filter_list(values: List[Any], idx_to_keep: Set[int]) -> List[Any]:
    return [x for i, x in enumerate(values) if i in idx_to_keep]


def filter_outliers(values: List[float]) -> List[float]:
    return list(filter(lambda x: x < 80, values))


def generate_random_sequence(mean: np.ndarray, std: np.ndarray, seq_length: int, rand: np.random.RandomState) -> np.ndarray:
    """
    Creates a [T, D] random sequence where each feature value is N(mean, Diag(std))

    Args:
        mean: A [D] vector containing the mean value
        std: A [D] vector containing the feature-wise standard deviation
    Returns:
        A [T, D] array representing the random sequence
    """
    elements_list: List[np.ndarray] = []

    num_features = mean.shape[0]
    for idx in range(num_features):
        features = rand.normal(loc=mean[idx], scale=std[idx], size=seq_length)  # [T]
        elements_list.append(np.expand_dims(features, axis=-1))

    return np.concatenate(elements_list, axis=-1)  # [T, D]


def get_num_non_exhausted(budget_per_seq: float, observed_per_seq: float, num_seq: int) -> int:
    """
    Gets the number of sequences that operate before budget exhaustion.

    Args:
        budget_per_seq: The energy per sequence as specified by the budget
        observed_per_seq: The observed energy per sequence
        num_seq: The number of sequences
    Returns:
        The number of sequences in which the system can compute before random guessing
    """
    budget = budget_per_seq * num_seq
    max_num_seq = int(budget / observed_per_seq)
    return max_num_seq


def get_random_guessing_errors(inputs: np.ndarray, max_num_seq: int, num_seq: int, mean: np.ndarray, std: np.ndarray, rand: np.random.RandomState) -> List[float]:
    """
    Gets the errors associated with random guessing after max_num_seq

    Args:
        inputs: The original sequence values
        max_num_seq: The number of sequences before exhausting the budget
        num_seq: The total number of sequences
        rand: The random number generator for stable results
    Returns:
        A list of [num_seq - max_num_seq] errors
    """
    error_list: List[float] = []
    seq_length = inputs.shape[1]

    for idx in range(max_num_seq, num_seq):
        seq = inputs[idx]
        guessed = generate_random_sequence(mean, std, seq_length=seq_length, rand=rand)
        error = np.average(np.abs(seq - guessed))
        error_list.append(error)

    return error_list


def enforce_budget(energy: List[float], errors: List[float], baseline_energy: List[float], inputs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> PolicyResult:
    """
    Enforces the budget on the given energy readings and adjusts the error accordingly.

    Args:
        energy: A list of energy values per sequences
        errors: A list of errors for each sequence
        baseline_energy: A list of baseline energy values (from which we get the budget)
    Returns:
        A tuple of the overall MAE and avg energy / seq (mJ)
    """
    rand = np.random.RandomState(832)
    num_seq = len(errors)

    # Remove outlier values
    energy = filter_outliers(energy)
    avg_energy = np.average(energy)

    baseline_energy = filter_outliers(baseline_energy)
    avg_budget = np.average(baseline_energy)

    # Compare the energy distributions
    t_stat, p_value = stats.ttest_ind(baseline_energy, energy, equal_var=False)
    p_value /= 2  # One-sided test

    if (t_stat > SMALL_NUMBER) or (p_value >= 0.05):
        return PolicyResult(energy=avg_energy,
                            energy_std=np.std(energy),
                            error=np.average(errors),
                            p_value=p_value,
                            t_stat=t_stat)

    # Enforce the energy budget through random guessing
    max_num_seq = get_num_non_exhausted(budget_per_seq=avg_budget,
                                        observed_per_seq=avg_energy,
                                        num_seq=num_seq)

    exhausted_errors = get_random_guessing_errors(inputs=inputs,
                                                  max_num_seq=max_num_seq,
                                                  num_seq=num_seq,
                                                  mean=mean,
                                                  std=std,
                                                  rand=rand)

    errors_with_budget = errors[:max_num_seq] + exhausted_errors
    assert len(errors_with_budget) == num_seq, 'Got {0} Errors. Expected: {1}'.format(len(errors_with_budget), num_seq)

    return PolicyResult(energy=avg_energy,
                        energy_std=np.std(energy),
                        error=np.average(errors_with_budget),
                        p_value=p_value,
                        t_stat=t_stat)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--num-trials', type=int, required=True)
    args = parser.parse_args()

    base = os.path.join('..', 'device', 'results', args.dataset)

    results: Dict[str, Dict[int, Summary]] = dict()

    for policy_folder in iterate_dir(base, pattern='.*'):
        policy_name = os.path.split(policy_folder)[-1]
        results[policy_name] = dict()

        for budget_folder in iterate_dir(policy_folder, pattern='.*'):
            # Read the energy summary
            energy_path = os.path.join(budget_folder, 'energy.json')
            if not os.path.exists(energy_path):
                continue

            energy_summary = read_json(energy_path)
            energy_list: List[float] = energy_summary['op_energy'][:75]
            comm_list: List[float] = energy_summary['comm_energy'][:75]

            # Get the collection rate
            try:
                collection_rate = int(os.path.split(budget_folder)[-1])
            except ValueError:
                continue

            # Read the error logs
            mae_list: List[float] = []
            bytes_list: List[float] = []
            num_collected = 0

            for trial in range(args.num_trials):
                error_log_path = os.path.join(budget_folder, '{0}_{1}_trial{2}.json.gz'.format(policy_name.replace('padded', 'standard'), collection_rate, trial))

                if not os.path.exists(error_log_path):
                    continue

                error_log = read_json_gz(error_log_path)
                maes = error_log['maes']
                num_bytes = error_log['num_bytes']

                mae_list.extend(maes)
                bytes_list.extend(num_bytes)
                num_collected += error_log['recv_count']

            if len(mae_list) > 0:
                results[policy_name][collection_rate] = Summary(mae=mae_list,
                                                                energy=energy_list,
                                                                comm=comm_list,
                                                                num_collected=num_collected,
                                                                num_bytes=bytes_list)

        # Remove any empty results
        if len(results[policy_name]) == 0:
            del results[policy_name]


    # Load the data and mean / std
    inputs, _ = load_data(dataset_name=args.dataset, fold='test')
    distribution = read_json(os.path.join('..', 'datasets', args.dataset, 'distribution.json'))
    mean, std = np.array(distribution['mean']), np.array(distribution['std'])

    collection_rates = list(sorted(results['uniform_standard'].keys()))
    
    for rate in collection_rates:
        idx_to_keep = get_overlapping_indices(policy_results=results, rate=rate)

        # Use the uniform policy to get the budget
        baseline_energy = results['uniform_standard'][rate].energy
        num_seq = len(baseline_energy)

        baseline_energy = filter_list(baseline_energy, idx_to_keep)
        budget_energy = filter_outliers(baseline_energy)
        budget = np.average(budget_energy) * num_seq

        print('Collection Rate: {0}, Budget: {1:.5f}mJ'.format(rate, budget))

        for policy_name, policy_result in results.items():
            energy = filter_list(policy_result[rate].energy, idx_to_keep)
            errors = filter_list(policy_result[rate].mae, idx_to_keep)
            num_bytes = policy_result[rate].num_bytes

            budget_result = enforce_budget(energy=energy,
                                           errors=errors,
                                           baseline_energy=baseline_energy,
                                           mean=mean,
                                           std=std,
                                           inputs=inputs)

            avg_bytes = np.average(num_bytes)
            std_bytes = np.std(num_bytes)

            print('{0} & {1:.5f} & {2:.2f} ($\\pm{3:.2f}$) & {4:.2f} ($\\pm {5:.2f}$) & {6:.4f}'.format(policy_name, budget_result.error, budget_result.energy, budget_result.energy_std, avg_bytes, std_bytes, budget_result.p_value))

        print()
