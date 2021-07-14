import numpy as np
import math
import scipy.stats as stats
from argparse import ArgumentParser
from collections import Counter, defaultdict, OrderedDict, namedtuple
from typing import Dict, List, DefaultDict, Optional, Tuple, Iterable

from adaptiveleak.analysis.plot_utils import iterate_policy_folders
from adaptiveleak.utils.constants import POLICIES, SMALL_NUMBER
from adaptiveleak.utils.file_utils import read_json_gz, iterate_dir, save_json_gz


JointKey = namedtuple('JointKey', ['label', 'size'])


def get_label_distribution(byte_dist: Dict[int, List[int]]) -> Dict[int, float]:
    """
    Finds the empirical frequency of each label.

    Args:
        byte_dist: A dictionary mapping label -> list of message sizes (in bytes)
    Returns:
        A dictionary mapping the label to its frequency in the data.
    """
    counts: Dict[int, int] = dict()
    total = 0

    for label, sizes in byte_dist.items():
        counts[label] = len(sizes)
        total += len(sizes)

    return {label: count / total for label, count in counts.items()}


def get_size_distribution(byte_dist: Dict[int, List[int]]) -> Dict[int, float]:
    """
    Finds the empirical frequency of each message size.

    Args:
        byte_dict: A dictionary mapping label -> list of message sizes (in bytes)
    Returns:
        A dictionary mapping the message size to its frequency in the data.
    """
    counts: Counter = Counter()
    total = 0

    for sizes in byte_dist.values():
        for size in sizes:
            counts[size] += 1

        total += len(sizes)

    return {size: count / total for size, count in counts.items()}


def get_joint_distribution(byte_dist: Dict[int, List[int]]) -> Dict[JointKey, float]:
    """
    Forms the empirical joint distribution between message size and label.

    Args:
        byte_dist: A dictionary mapping label -> list of message sizes (in bytes)
    Returns:
        A dictionary with key (label, size) and value equal to the frequency in the data.
    """
    counts: Counter = Counter()
    total = 0

    for label, sizes in byte_dist.items():
        for size in sizes:
            key = JointKey(label=label, size=size)
            counts[key] += 1
            total += 1

    return {key: count / total for key, count in counts.items()}


def mutual_information(byte_dist: Dict[int, List[int]]) -> float:
    # Compute the required empirical distributions
    label_dist = get_label_distribution(byte_dist=byte_dist)
    size_dist = get_size_distribution(byte_dist=byte_dist)
    joint_dist = get_joint_distribution(byte_dist=byte_dist)

    # Calculate the mutual information
    information = 0.0

    for label, size in joint_dist.keys():
        # Unpack the required probabilities
        label_prob = label_dist[label]
        size_prob = size_dist[size]

        key = JointKey(label=label, size=size)
        joint_prob = joint_dist[key]

        information += joint_prob * np.log((joint_prob) / (label_prob * size_prob + SMALL_NUMBER))

    # Return the mutual information
    return information


def randomize_distribution(byte_dist: Dict[int, List[int]], rand: np.random.RandomState, num_trials: int) -> Iterable[DefaultDict[int, List[int]]]:
    """
    Creates random permutations of the given message size distribution.
    """
    # Flatten the data
    labels: List[int] = []
    sizes: List[int] = []

    for label, label_sizes in byte_dist.items():
        for size in label_sizes:
            labels.append(label)
            sizes.append(size)

    for _ in range(num_trials):
        # Shuffle the sizes
        rand.shuffle(sizes)

        # Create the new distribution
        perm_dist: DefaultDict[int, List[int]] = defaultdict(list)
        
        for label, size in zip(labels, sizes):
            perm_dist[label].append(size)

        yield perm_dist


def compute_p_value(information: float, randomized_information: List[float]) -> float:
    num_greater_eq = sum(int(r >= information) for r in randomized_information)
    return (num_greater_eq + 1) / (len(randomized_information) + 1)


def run_test(byte_dist: Dict[int, List[int]], num_trials: int) -> Tuple[float, float]:
    """
    Runs a Chi-Square test to analyze the byte distribution.

    Args:
        byte_dist: A map of label -> List of message sizes (in bytes)
    """
    # Get the mutual information of the existing distribution
    information = mutual_information(byte_dist)

    # Get the mutual information of the randomized distributions
    randomized_information: List[float] = []
    rand = np.random.RandomState(seed=23634)

    for perm_dist in randomize_distribution(byte_dist, rand=rand, num_trials=num_trials):
        randomized_information.append(mutual_information(perm_dist))

    # Get the percentile of the observed mutual information within the randomized distribution
    perc = stats.percentileofscore(randomized_information, information)
    p_value = compute_p_value(information=information, randomized_information=randomized_information)

    return {
        'percentile': perc,
        'p_value': p_value,
        'randomized_mi': { 'avg': float(np.average(randomized_information)), 'std': float(np.std(randomized_information)) },
        'mutual_information': float(information),
        'num_trials': num_trials
    }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dates', type=str, required=True, nargs='+')
    parser.add_argument('--precision', type=float, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    assert args.precision > 0, 'Precision must be positive'

    # Compute the number of trials based on a conservative upper bound
    num_trials = int(math.ceil(1.0 / (4 * args.precision**2)))

    information_results: DefaultDict[str, Dict[float, float]] = defaultdict(dict)

    for folder in iterate_policy_folders(args.dates, dataset=args.dataset):
        for sim_file in iterate_dir(folder, pattern='.*json.gz'):
            model = read_json_gz(sim_file)

            num_bytes = model['num_bytes']
            labels = model['labels']

            byte_dist: DefaultDict[int, List[int]] = defaultdict(list)
            for label, byte_count in zip(labels, num_bytes):
                byte_dist[label].append(byte_count)

            name = '{0}_{1}'.format(model['policy']['policy_name'].lower(), model['policy']['encoding_mode'].lower())
            energy_per_seq = model['policy']['energy_per_seq']

            if name not in ('adaptive_heuristic_standard', 'adaptive_deviation_standard', 'skip_rnn_standard'):
                test_result = run_test(byte_dist, num_trials=1)
            else:
                test_result = run_test(byte_dist, num_trials=num_trials)

            model['mututal_information'] = test_result
            save_json_gz(model, sim_file)
