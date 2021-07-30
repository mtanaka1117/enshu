import matplotlib.pyplot as plt
import os
import numpy as np
from argparse import ArgumentParser
from functools import partial
from scipy import stats
from collections import namedtuple, OrderedDict
from typing import Any, Dict, List, Optional

from adaptiveleak.utils.constants import POLICIES, ENCODING, SMALL_NUMBER
from adaptiveleak.utils.file_utils import read_json_gz
from adaptiveleak.analysis.plot_utils import COLORS, to_label, geometric_mean, MARKER, MARKER_SIZE, LINE_WIDTH, PLOT_STYLE
from adaptiveleak.analysis.plot_utils import PLOT_SIZE, AXIS_FONT, LEGEND_FONT, TITLE_FONT
from adaptiveleak.analysis.plot_utils import extract_results, iterate_policy_folders, dataset_label


AttackResult = namedtuple('AttackResult', ['median', 'first', 'third', 'raw'])


def get_attack_results(date: str, dataset: str) -> Dict[str, AttackResult]:
    extract_fn = partial(extract_results, field='attack', aggregate_mode=None)
    policy_folders = list(iterate_policy_folders([date], dataset=dataset))

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
                                           raw=accuracy)

    return result


def plot(dataset_results: Dict[str, Dict[str, AttackResult]], output_file: Optional[str], is_group_comp: bool):

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(PLOT_SIZE[0] * 1.5, PLOT_SIZE[1]))
        #fig.subplots_adjust(right=0.8)

        labels: List[str] = []
        agg_errors: List[float] = []

        policy_names = ['adaptive_heuristic', 'adaptive_deviation'] if is_group_comp else POLICIES
        encoding_names = ['single_group', 'group_unshifted', 'pruned', 'group'] if is_group_comp else ['standard', 'padded', 'group']

        width = 0.15
        offset = -1 * width * 2.5

        xs = np.arange(len(dataset_results) + 1)  # Include the 'All'

        # Print the label for the 'Overall' table
        ax.text(3.5, 80 - 25 * (offset - width), 'Overall Medians', fontweight='bold', fontsize=LEGEND_FONT)

        for name in policy_names:
            encodings = encoding_names if name not in ('uniform', 'random') else ['standard']

            for encoding in encodings:
                policy_name = '{0}_{1}'.format(name, encoding)

                median_errors: List[float] = []
                first_errors: List[float] = []
                third_errors: List[float] = []
                raw_errors: List[float] = []

                for dataset, policy_results in sorted(dataset_results.items()):
                    if (policy_name not in policy_results):
                        print('{0}: {1}'.format(dataset, policy_name))
                        continue

                    median_errors.append(policy_results[policy_name].median)
                    first_errors.append(policy_results[policy_name].median - policy_results[policy_name].first)
                    third_errors.append(policy_results[policy_name].third - policy_results[policy_name].median)
                    raw_errors.extend(policy_results[policy_name].raw)

                if len(median_errors) == (len(xs) - 1):
                    aggregate = np.median(raw_errors)

                    median_errors.append(aggregate)
                    first_errors.append(aggregate - np.percentile(raw_errors, 25))
                    third_errors.append(np.percentile(raw_errors, 75) - aggregate)

                    label_name = name if encoding == 'standard' else policy_name
                    ax.bar(xs + offset, median_errors, width=width, color=COLORS[policy_name], label=to_label(label_name))
                    ax.errorbar(xs + offset, median_errors, yerr=[first_errors, third_errors], color='k', capsize=2, ls='none')

                    # Annotate the aggregate score
                    ax.text(3.5, 80 - 25 * offset, '{0}: {1:.2f}%'.format(to_label(label_name), aggregate), fontsize=LEGEND_FONT)

                offset += width

        dataset_names = [dataset for dataset in sorted(dataset_results.keys())]
        dataset_names.append('Overall')

        ax.set_xticks(xs)
        ax.set_xticklabels(dataset_names)

        # Add a line to separate the 'All' category
        ax.axvline((xs[-1] + xs[-2]) / 2, linestyle='--', color='k')

        ax.set_xlabel('Dataset', fontsize=AXIS_FONT)
        ax.set_ylabel('Median Attack Accuracy (%)', fontsize=AXIS_FONT)
        ax.set_title('Median Attack Accuracy on All Datasets', fontsize=TITLE_FONT)

        ax.legend(fontsize=LEGEND_FONT)

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file, bbox_inches='tight', transparent=True)
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--date', type=str, required=True)
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--is-group-comp', action='store_true')
    args = parser.parse_args()

    print('Num Datasets: {0}'.format(len(args.datasets)))
    print('==========')

    dataset_errors: Dict[str, Dict[str, float]] = dict()

    for dataset in args.datasets:
        dataset_errors[dataset_label(dataset)] = get_attack_results(date=args.date, dataset=dataset)

    plot(dataset_errors, output_file=args.output_file, is_group_comp=args.is_group_comp)
