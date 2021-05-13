import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import Counter, defaultdict, OrderedDict
from typing import Dict, List, DefaultDict, Optional, Tuple

from adaptiveleak.analysis.plot_utils import COLORS, PLOT_STYLE, LINE_WIDTH, MARKER, MARKER_SIZE, to_label
from adaptiveleak.analysis.plot_utils import LEGEND_FONT, AXIS_FONT, PLOT_SIZE, TITLE_FONT
from adaptiveleak.analysis.plot_utils import iterate_policy_folders, dataset_label
from adaptiveleak.utils.constants import POLICIES, SMALL_NUMBER
from adaptiveleak.utils.file_utils import read_json_gz, iterate_dir



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


def get_joint_distribution(byte_dist: Dict[int, List[int]]) -> Dict[Tuple[int, int], float]:
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
            key = (label, size)
            counts[key] += 1
            total += 1

    return {key: count / total for key, count in counts.items()}



def get_conditional_distribution(byte_dist: Dict[int, List[int]]) -> Dict[int, Dict[int, float]]:
    size_dist: DefaultDict[int, Dict[int, int]] = defaultdict(dict)

    for label, sizes in byte_dist.items():
        for size in sizes:
            if label not in size_dist[size]:
                size_dist[size][label] = 0

            size_dist[size][label] += 1
    
    # Normalize the results
    result: Dict[int, Dict[int, float]] = dict()
    for size, label_counts in size_dist.items():
        total = sum(label_counts.values())
        result[size] = {label: count / total for label, count in label_counts.items()}

    return result


def calculate_entropy(distribution: Dict[int, float]) -> float:
    """
    Calculates the Shannon entropy in the given distribution.

    Args:
        distribution: A dictionary mapping value -> freq
    Returns:
        The empirical entropy in nits.
    """
    return sum(map(lambda x: -1 * x * np.log(x + SMALL_NUMBER), distribution.values())) 


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
        joint_prob = joint_dist[(label, size)]

        information += joint_prob * np.log((joint_prob) / (label_prob * size_prob + SMALL_NUMBER))

    # Return the mutual information
    return information


def plot(information_results: DefaultDict[str, Dict[float, float]], dataset: str, output_file: Optional[str]):
    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots(figsize=PLOT_SIZE)

        for name in POLICIES:
            if name not in information_results:
                continue

            information = information_results[name]

            fractions = sorted(information.keys())
            values = [information[frac] for frac in fractions]

            ax.plot(fractions, values, label=to_label(name), color=COLORS[name], linewidth=LINE_WIDTH, marker=MARKER, markersize=MARKER_SIZE)

            print('{0} & {1:.2f} ({2:.2f})'.format(name, np.average(values), np.max(values)))

        ax.legend(fontsize=LEGEND_FONT, loc='center')

        ax.set_title('Empirical Mutual Information between Message Size and Label on the {0} Dataset'.format(dataset_label(dataset)), fontsize=TITLE_FONT)
        ax.set_xlabel('Target Fraction', fontsize=AXIS_FONT)
        ax.set_ylabel('Empirical Mutual Information (nits)', fontsize=AXIS_FONT)

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dates', type=str, required=True, nargs='+')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    information_results: DefaultDict[str, Dict[float, float]] = defaultdict(dict)

    for folder in iterate_policy_folders(args.dates, dataset=args.dataset):
        for sim_file in iterate_dir(folder, pattern='.*json.gz'):
            model = read_json_gz(sim_file)

            num_bytes = model['num_bytes']
            labels = model['labels']

            byte_dist: DefaultDict[int, List[int]] = defaultdict(list)
            for label, byte_count in zip(labels, num_bytes):
                byte_dist[label].append(byte_count)

            name = model['policy']['name']
            target = model['policy']['target']

            information_results[name][target] = mutual_information(byte_dist)

    plot(information_results, dataset=args.dataset, output_file=args.output_file)
