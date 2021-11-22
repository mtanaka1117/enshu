import numpy as np
import math
import scipy.stats as stats
import time
from argparse import ArgumentParser
from sklearn.metrics import mutual_info_score
from collections import Counter, defaultdict, OrderedDict, namedtuple
from typing import Dict, List, DefaultDict, Optional, Tuple, Iterable

from adaptiveleak.analysis.plot_utils import iterate_policy_folders
from adaptiveleak.utils.constants import POLICIES, SMALL_NUMBER
from adaptiveleak.utils.file_utils import read_json_gz, iterate_dir, save_json_gz


BIN_FACTOR = 10


def compute_mutual_information(labels: List[int], byte_counts: List[int]) -> float:
    # Compute the joint distribution via a histogram
    num_samples = len(labels)
    byte_bins = int(num_samples / BIN_FACTOR)
    label_bins = np.amax(labels) + 1

    joint_hist, _, _ = np.histogram2d(labels, byte_counts, bins=(label_bins, byte_bins))

    # Compute the empirical mutual information
    mutual_information = mutual_info_score(None, None, contingency=joint_hist)

    return mutual_information


def randomize_distribution(sizes: List[int], rand: np.random.RandomState, num_trials: int) -> Iterable[List[int]]:
    """
    Creates random permutations of the given message size distribution.
    """
    for _ in range(num_trials):
        # Shuffle the sizes
        rand.shuffle(sizes)

        # Create the new distribution
        yield sizes


def compute_p_value(information: float, randomized_information: List[float]) -> float:
    num_greater_eq = sum(int(r >= information) for r in randomized_information)
    return (num_greater_eq + 1) / (len(randomized_information) + 1)


def compute_entropy(counts: List[int]) -> float:
    probs = counts / np.sum(counts)
    return -1 * sum((p * np.log(p)) for p in probs if p > SMALL_NUMBER)


def run_test(byte_dist: Dict[int, List[int]], num_trials: int) -> Tuple[float, float]:
    """
    Runs a Chi-Square test to analyze the byte distribution.

    Args:
        byte_dist: A map of label -> List of message sizes (in bytes)
    """
    # Extract a list of labels and byte counts
    labels: List[int] = []
    byte_counts: List[int] = []

    for label, counts in byte_dist.items():
        for count in counts:
            labels.append(label)
            byte_counts.append(count)

    # Compute the entropy of each distribution
    label_counts = np.bincount(labels).astype(float)
    label_entropy = compute_entropy(label_counts)

    num_bins = int(len(labels) / BIN_FACTOR)
    byte_hist, _ = np.histogram(byte_counts, bins=num_bins)
    byte_entropy = compute_entropy(byte_hist)

    # Get the (normalized) mutual information of the existing distribution
    information = compute_mutual_information(labels, byte_counts)
    norm_information = (2 * information) / (label_entropy + byte_entropy)

    # Get the mutual information of the randomized distributions
    randomized_information: List[float] = []
    rand = np.random.RandomState(seed=23634)

    for perm_sizes in randomize_distribution(byte_counts, rand=rand, num_trials=num_trials):
        perm_information = compute_mutual_information(labels, perm_sizes)
        norm_perm_information = (2 * perm_information) / (label_entropy + byte_entropy)

        randomized_information.append(norm_perm_information)

    # Get the percentile of the observed mutual information within the randomized distribution
    perc = stats.percentileofscore(randomized_information, norm_information)
    p_value = compute_p_value(information=norm_information, randomized_information=randomized_information)

    return {
        'percentile': perc,
        'p_value': p_value,
        'randomized_mi': {'avg': float(np.average(randomized_information)), 'std': float(np.std(randomized_information))},
        'norm_mutual_information': float(norm_information),
        'num_trials': num_trials
    }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='The name of the experiment log folder.')
    parser.add_argument('--trials', type=int, required=True, help='The number of trials to execute for the permutation test.')
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset.')
    args = parser.parse_args()

    assert args.trials > 0, '# Trials must be positive'

    information_results: DefaultDict[str, Dict[float, float]] = defaultdict(dict)

    for folder in iterate_policy_folders([args.folder], dataset=args.dataset):
        for sim_file in iterate_dir(folder, pattern='.*json.gz'):
            model = read_json_gz(sim_file)

            num_bytes = model['num_bytes']
            labels = model['labels']

            byte_dist: DefaultDict[int, List[int]] = defaultdict(list)
            for label, byte_count in zip(labels, num_bytes):
                byte_dist[label].append(byte_count)

            encoding_mode = model['policy']['encoding_mode'].lower()

            if encoding_mode in ('single_group', 'group_unshifted'):
                continue

            name = '{0}_{1}'.format(model['policy']['policy_name'].lower(), encoding_mode)
            energy_per_seq = model['policy']['energy_per_seq']

            if name not in ('adaptive_heuristic_standard', 'adaptive_deviation_standard', 'skip_rnn_standard'):
                test_result = run_test(byte_dist, num_trials=1)
            else:
                test_result = run_test(byte_dist, num_trials=args.trials)

            model['mutual_information'] = test_result
            save_json_gz(model, sim_file)
