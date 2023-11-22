"""
Plots the aggregate attack results for all datasets.
"""
import matplotlib.pyplot as plt
import os
import numpy as np
from argparse import ArgumentParser
from functools import partial
from collections import namedtuple, OrderedDict
from typing import Any, Dict, List, Optional

from adaptiveleak.utils.constants import POLICIES, ENCODING, SMALL_NUMBER
from adaptiveleak.utils.file_utils import read_json_gz
from adaptiveleak.analysis.plot_utils import COLORS, to_label, geometric_mean, MARKER, MARKER_SIZE, LINE_WIDTH, PLOT_STYLE, ANNOTATE_FONT
from adaptiveleak.analysis.plot_utils import PLOT_SIZE, AXIS_FONT, LEGEND_FONT, TITLE_FONT
from adaptiveleak.analysis.plot_utils import extract_results, iterate_policy_folders, dataset_label


AttackResult = namedtuple('AttackResult', ['median', 'first', 'third', 'raw', 'minimum', 'maximum'])

DATASET_LABELS = {
    'uci_har': 'Activity',
    'trajectories': 'Chars.',
    'eog': 'EOG',
    'epilepsy': 'Epilepsy',
    'mnist': 'MNIST',
    'haptics': 'Pswd.',
    'pavement': 'Pavem.',
    'strawberry': 'Strawb.',
    'tiselac': 'Tiselac'
}


def get_attack_results(folder: str, dataset: str) -> Dict[str, AttackResult]:
    extract_fn = partial(extract_results, field='attack', aggregate_mode=None)
    policy_folders = list(iterate_policy_folders([folder], dataset=dataset))

    sim_results = {name: res for name, res in map(extract_fn, policy_folders)}

    # Standardize which energy budgets we will use to prevent bugs
    baseline_results = sim_results['uniform_standard']
    energy_budgets = list(sorted(baseline_results.keys()))

    result: Dict[str, AttackResult] = dict()

    for policy_name, policy_results in sim_results.items():
        if (len(policy_results) == 0) or any(isinstance(policy_results[b], float) for b in energy_budgets):
            continue

        accuracy: List[float] = []
        for b in energy_budgets:
            accuracy.extend((x * 100 for x in policy_results[b]['test']['accuracy']))

        result[policy_name] = AttackResult(median=np.median(accuracy),
                                           first=np.percentile(accuracy, 25),
                                           third=np.percentile(accuracy, 75),
                                           minimum=np.min(accuracy),
                                           maximum=np.max(accuracy),
                                           raw=accuracy)

    return result


def plot(dataset_results: Dict[str, Dict[str, AttackResult]], output_file: Optional[str]):

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(PLOT_SIZE[0] * 1.25, PLOT_SIZE[1] * 0.75))

        labels: List[str] = []
        agg_errors: List[float] = []

        policy_names = ['adaptive_heuristic', 'adaptive_deviation']
        encoding_names = ['standard', 'group']

        width = 0.2
        offset = -1 * width * 1.5

        xs = np.arange(len(dataset_results) + 1)  # Include the 'All'

        for name in policy_names:
            encodings = encoding_names if name not in ('uniform', 'random') else ['standard']

            for encoding in encodings:
                policy_name = '{0}_{1}'.format(name, encoding)

                median_errors: List[float] = []
                first_errors: List[float] = []
                third_errors: List[float] = []
                raw_errors: List[float] = []
                max_errors: List[float] = []

                for dataset, policy_results in sorted(dataset_results.items()):
                    if (policy_name not in policy_results):
                        continue

                    median_errors.append(policy_results[policy_name].median)
                    first_errors.append(policy_results[policy_name].median - policy_results[policy_name].first)
                    third_errors.append(policy_results[policy_name].third - policy_results[policy_name].median)
                    max_errors.append(policy_results[policy_name].maximum)
                    raw_errors.extend(policy_results[policy_name].raw)

                if len(median_errors) == (len(xs) - 1):
                    aggregate = np.median(raw_errors)

                    median_errors.append(aggregate)
                    first_errors.append(aggregate - np.percentile(raw_errors, 25))
                    third_errors.append(np.percentile(raw_errors, 75) - aggregate)
                    max_errors.append(np.max(raw_errors))

                    label_name = name if encoding == 'standard' else policy_name
                    ax.bar(xs + offset, median_errors, width=width, color=COLORS[policy_name], label=to_label(label_name))
                    ax.errorbar(xs + offset, median_errors, yerr=[first_errors, third_errors], color='k', capsize=2, ls='none')
                    ax.scatter(xs + offset, max_errors, color=COLORS[policy_name])

                offset += width

        dataset_names = [dataset for dataset in sorted(dataset_results.keys())]
        dataset_names.append('Overall')

        ax.set_xticks(xs)
        ax.set_xticklabels(dataset_names, fontsize=AXIS_FONT - 3)
        ax.set_yticklabels([round(y, 3) for y in ax.get_yticks()], fontsize=AXIS_FONT)

        # Add a line to separate the 'All' category
        ax.axvline((xs[-1] + xs[-2]) / 2, linestyle='--', color='k')

        ax.set_xlabel('Dataset', fontsize=AXIS_FONT)
        ax.set_ylabel('Median Accuracy (%)', fontsize=AXIS_FONT)
        ax.set_title('Attacker Event Detection Accuracy', fontsize=TITLE_FONT)

        ax.legend(fontsize=LEGEND_FONT)

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file, bbox_inches='tight', transparent=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Name of the experiment log folder.')
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='A list of dataset names.')
    parser.add_argument('--output-file', type=str, help='An optional path to an output file.')
    args = parser.parse_args()

    print('Num Datasets: {0}'.format(len(args.datasets)))

    dataset_errors: Dict[str, Dict[str, float]] = dict()

    for dataset in args.datasets:
        dataset_errors[DATASET_LABELS[dataset]] = get_attack_results(folder=args.folder, dataset=dataset)

    plot(dataset_errors, output_file=args.output_file)
