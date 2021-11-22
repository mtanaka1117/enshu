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
LabelPair = namedtuple('LabelPair', ['first', 'second'])


def compute_mutual_information(labels: List[int], byte_counts: List[int]) -> float:
    # Compute the joint distribution via a histogram
    num_samples = len(labels)
    byte_bins = int(num_samples / BIN_FACTOR)
    label_bins = np.amax(labels) + 1

    joint_hist, _, _ = np.histogram2d(labels, byte_counts, bins=(label_bins, byte_bins))

    # Compute the empirical mutual information
    mutual_information = mutual_info_score(None, None, contingency=joint_hist)

    return mutual_information


def compute_entropy(counts: List[int]) -> float:
    probs = counts / np.sum(counts)
    return -1 * sum((p * np.log(p)) for p in probs if p > SMALL_NUMBER)


def run(byte_dist: Dict[int, List[int]]) -> Tuple[float, float]:
    """
    Runs a Chi-Square test to analyze the byte distribution.

    Args:
        byte_dist: A map of label -> List of message sizes (in bytes)
    """
    # Create a map from re-named labels to label pairs
    idx = 0
    label_map: Dict[LabelPair, int] = dict()

    for label_one in byte_dist.keys():
        for label_two in byte_dist.keys():
            if label_one <= label_two:
                pair = LabelPair(first=label_one, second=label_two)
                label_map[pair] = idx
                idx += 1

    # Extract a list of labels and byte counts
    labels: List[int] = []
    byte_counts: List[int] = []

    rand = np.random.RandomState(seed=12309)

    for label_one, counts_one in byte_dist.items():
        for label_two, counts_two in byte_dist.items():
            if label_one > label_two:
                continue

            size = len(counts_one) + len(counts_two)
            rand_counts_one = rand.choice(counts_one, replace=True, size=size)
            rand_counts_two = rand.choice(counts_two, replace=True, size=size)

            total_counts = rand_counts_one + rand_counts_two
            key = LabelPair(first=label_one, second=label_two)

            for count in total_counts:
                labels.append(label_map[key])
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

    print(norm_information)

#    return {
#        'percentile': perc,
#        'p_value': p_value,
#        'randomized_mi': { 'avg': float(np.average(randomized_information)), 'std': float(np.std(randomized_information)) },
#        'norm_mutual_information': float(norm_information),
#        'num_trials': num_trials
#    }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dates', type=str, required=True, nargs='+')
    parser.add_argument('--trials', type=int, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    assert args.trials > 0, '# Trials must be positive'

    information_results: DefaultDict[str, Dict[float, float]] = defaultdict(dict)

    for folder in iterate_policy_folders(args.dates, dataset=args.dataset):
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

            run(byte_dist)

            #model['mutual_information'] = test_result
            #save_json_gz(model, sim_file)
