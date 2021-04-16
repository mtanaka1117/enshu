import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import Counter, defaultdict, OrderedDict
from typing import Dict, List, DefaultDict, Optional

from adaptiveleak.plotting.plot_utils import COLORS, PLOT_STYLE, LINE_WIDTH, MARKER, MARKER_SIZE, to_label
from adaptiveleak.plotting.plot_utils import LEGEND_FONT, AXIS_FONT, PLOT_SIZE, TITLE_FONT
from adaptiveleak.utils.constants import POLICIES, SMALL_NUMBER
from adaptiveleak.utils.file_utils import read_json_gz, iterate_dir



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

            if cond_prob > SMALL_NUMBER:
                cond_entropy += -1 * (size_prob * cond_prob) * np.log(cond_prob)

    return label_entropy - cond_entropy


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

            print('{0} & {1:.4f}'.format(name, np.max(values)))

        ax.legend(fontsize=LEGEND_FONT)

        ax.set_title('Empirical Mutual Information between Message Size and Label on the {0} Dataset'.format(args.dataset.capitalize()), fontsize=TITLE_FONT)
        ax.set_xlabel('Target Fraction', fontsize=AXIS_FONT)
        ax.set_ylabel('Empirical Mutual Information (nits)', fontsize=AXIS_FONT)

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--policy-folders', type=str, required=True, nargs='+')
    parser.add_argument('--dataset', type=str, required=True)
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

    plot(information_results, dataset=args.dataset, output_file=args.output_file)
