import numpy as np
import os.path
import math
from argparse import ArgumentParser
from sklearn.metrics import mutual_info_score
from collections import namedtuple
from typing import List, Set, Any

from adaptiveleak.utils.constants import SMALL_NUMBER
from adaptiveleak.utils.file_utils import read_json_gz, iterate_dir, read_json


Summary = namedtuple('Summary', ['energy', 'comm', 'mae', 'num_collected', 'num_bytes'])
PolicyResult = namedtuple('PolicyResult', ['energy', 'energy_std', 'error', 'p_value', 't_stat'])

BIN_FACTOR = 10


def get_idx_to_keep(errors: List[float], recv_count: int, total_count: int) -> Set[int]:
    if recv_count < total_count:
        indices_to_keep = np.argsort(errors)[:recv_count]
    else:
        indices_to_keep = list(range(total_count))

    return set(indices_to_keep)


def filter_list(values: List[Any], idx_to_keep: Set[int]) -> List[Any]:
    return [x for i, x in enumerate(values) if i in idx_to_keep]


def compute_mutual_information(labels: List[int], byte_counts: List[int]) -> float:
    # Compute the joint distribution via a histogram
    num_samples = len(labels)
    byte_bins = int(num_samples / BIN_FACTOR)
    label_bins = np.amax(labels) + 1

    joint_hist, _, _ = np.histogram2d(labels, byte_counts, bins=(label_bins, byte_bins))

    # Compute the empirical mutual information
    mutual_information = mutual_info_score(None, None, contingency=joint_hist)

    return float(mutual_information)


def compute_entropy(counts: List[int]) -> float:
    probs = counts / np.sum(counts)
    return -1 * sum((p * np.log(p)) for p in probs if p > SMALL_NUMBER)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    base = os.path.join('..', 'device', 'results', args.dataset)

    for policy_folder in iterate_dir(base, pattern='.*'):
        policy_name = os.path.split(policy_folder)[-1]
        information_scores: List[float] = []

        for budget_folder in iterate_dir(policy_folder, pattern='.*'):
            # Read the energy summary
            energy_path = os.path.join(budget_folder, 'energy.json')
            if not os.path.exists(energy_path):
                continue

            # Get the collection rate
            try:
                collection_rate = int(os.path.split(budget_folder)[-1])
            except ValueError:
                continue

            # Read the error logs
            error_log_path = os.path.join(budget_folder, '{0}_{1}_trial0.json.gz'.format(policy_name.replace('padded', 'standard'), collection_rate))

            error_log = read_json_gz(error_log_path)
            recv_count = error_log['recv_count']

            idx_to_keep = get_idx_to_keep(errors=error_log['maes'], recv_count=recv_count, total_count=error_log['count'])

            num_bytes = error_log['num_bytes']
            labels = filter_list(error_log['labels'], idx_to_keep)

            mut_info = compute_mutual_information(labels=labels, byte_counts=num_bytes)
            label_entropy = compute_entropy(labels)
            bytes_entropy = compute_entropy(num_bytes)

            nmi = (mut_info) / (label_entropy + bytes_entropy)
            information_scores.append(nmi)

        print('{0}: {1}, {2}'.format(policy_name, np.median(information_scores), np.max(information_scores)))
