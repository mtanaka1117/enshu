import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import Counter, defaultdict, OrderedDict
from typing import Dict, List, DefaultDict, Optional

from utils.file_utils import read_json_gz, iterate_dir


MODEL_ORDER = ['random', 'uniform', 'adaptive_standard', 'adaptive_group']

COLORS = {
    'random': '#d73027',
    'uniform': '#fc8d59',
    'adaptive_standard': '#9ecae1',
    'adaptive_group': '#08519c'
}

#MODEL_ORDER = ['Random', 'Adaptive Heuristic', 'Adaptive Quantile']
#
#COLORS = {
#    'Random': '#66c2a5',
#    'Adaptive Heuristic': '#fc8d62',
#    'Adaptive Quantile': '#8da0cb'
#}



def get_label_distribution(byte_dist: Dict[int, List[int]]) -> Dict[int, float]:
    counts: Dict[int, int] = dict()
    total = 0

    for label, sizes in byte_dist.items():
        counts[label] = len(sizes)
        total += len(sizes)

    return {label: count / total for label, count in counts.items()}


def get_size_distribution(byte_dist: Dict[int, List[int]]) -> Dict[int, float]:
    counts: Counter = Counter()
    total = 0

    for sizes in byte_dist.values():
        for size in sizes:
            counts[size] += 1

        total += len(sizes)

    return {size: count / total for size, count in counts.items()}


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


def to_label(name: str) -> str:
    return ' '.join(t.capitalize() for t in name.split('_'))


def mutual_information(byte_dist: Dict[int, List[int]]) -> float:
    # Compute the required empirical distributions
    label_dist = get_label_distribution(byte_dist=byte_dist)
    cond_dist = get_conditional_distribution(byte_dist=byte_dist)
    size_dist = get_size_distribution(byte_dist=byte_dist)

    # Compute the label entropy
    label_entropy = 0
    for label, prob in label_dist.items():
        label_entropy += -1 * prob * np.log(prob)

    # Compute the conditional entropy
    cond_entropy = 0
    for size in size_dist.keys():
        for label in label_dist.keys():
            size_prob = size_dist[size]
            cond_prob = cond_dist[size].get(label, 0)

            if cond_prob > 1e-6:
                cond_entropy += -1 * (size_prob * cond_prob) * np.log(cond_prob)

    return label_entropy - cond_entropy



def get_name(policy: OrderedDict, is_padded: bool) -> str:
    name = policy['name'].capitalize()

    if name == 'Adaptive':
        compression = policy['compression_name'].capitalize()

        if compression == 'Fixed':
            return 'Adaptive'

        if not is_padded:
            if compression == 'Block':
                return '{0} Stream'.format(name)

        return '{0} {1}'.format(name, compression)

    return name


def plot(information_results: DefaultDict[str, Dict[float, float]], output_file: Optional[str]):
    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots()

        for name in MODEL_ORDER:
            if name not in information_results:
                continue

            information = information_results[name]

            fractions = sorted(information.keys())
            values = [information[frac] for frac in fractions]

            ax.plot(fractions, values, label=to_label(name), color=COLORS[name], linewidth=4, marker='o', markersize=8)

            print('{0} & {1:.4f}'.format(name, np.max(values)))

        ax.legend(fontsize=12)

        ax.set_title('Empirical Mutual Information between Message Size and Prediction')
        ax.set_xlabel('Target Fraction', size=12)
        ax.set_ylabel('Empirical Mutual Information (nits)', size=12)

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--policy-folders', type=str, required=True, nargs='+')
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    information_results: DefaultDict[str, Dict[float, float]] = defaultdict(dict)

    for folder in args.policy_folders:
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

    plot(information_results, output_file=args.output_file)
