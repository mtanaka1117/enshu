import matplotlib.pyplot as plt
import os
import numpy as np
from argparse import ArgumentParser
from functools import partial
from scipy import stats
from collections import namedtuple, OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, DefaultDict

from adaptiveleak.utils.constants import POLICIES, ENCODING
from adaptiveleak.utils.file_utils import read_json_gz
from adaptiveleak.analysis.plot_utils import COLORS, to_label, geometric_mean, MARKER, MARKER_SIZE, LINE_WIDTH, PLOT_STYLE
from adaptiveleak.analysis.plot_utils import PLOT_SIZE, AXIS_FONT, LEGEND_FONT, TITLE_FONT
from adaptiveleak.analysis.plot_utils import extract_results, iterate_policy_folders, dataset_label



def aggregate_for_collect_level(sim_results: Dict[str, Dict[str, Dict[float, float]]]) -> Dict[str, float]:
    model_results: DefaultDict[str, List[float]] = defaultdict(list)

    for dataset_name, dataset_results in sim_results.items():

        # Get the baseline results, list of error values
        uniform_results = dataset_results['uniform_standard']

        for policy_name, policy_results in dataset_results.items():
            
            # Compute the average normalized error for this data-set
            norm_errors = [(policy_results[r] - uniform_results[r]) for r in sorted(policy_results.keys())]
            avg_norm_error = np.average(norm_errors)

            model_results[policy_name].append(avg_norm_error)
   
    result: Dict[str, float] = dict()
    for policy_name, norm_errors in model_results.items():
        if 'unshifted' not in policy_name:
            result[policy_name] = np.average(norm_errors)

    return result


def plot(level_results: Dict[str, Dict[str, float]], levels: List[str], output_file: Optional[str]):

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=PLOT_SIZE)

        xs = list(range(len(levels)))

        for policy_name, policy_results in level_results.items():
            norm_errors: List[float] = [level_results[policy_name][level] for level in levels]
            ax.plot(xs, norm_errors, marker=MARKER, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=to_label(policy_name), color=COLORS[policy_name])

        #print(' & '.join(labels))
        #print(' & '.join((('{0:.5f}'.format(x) if i != min_error else '\\textbf{{{0:.5f}}}'.format(x)) for i, x in enumerate(agg_errors))))

        ax.set_xlabel('Collect Energy Level', fontsize=AXIS_FONT)
        ax.set_ylabel('Avg MAE Normalized to Uniform', fontsize=AXIS_FONT)
        ax.set_title('Average Reconstruction Error for Collection Energy Levels', fontsize=TITLE_FONT)

        ax.legend(fontsize=LEGEND_FONT)

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file, bbox_inches='tight')
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--levels', type=str, nargs='+', required=True)
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    extract_fn = partial(extract_results, field='mae', aggregate_mode=None)

    level_results: DefaultDict[str, Dict[str, float]] = defaultdict(dict)

    for level in args.levels:
        dataset_results: Dict[str, Dict[str, Dict[float, float]]] = dict()
        
        for dataset_name in args.datasets:
            policy_folders = list(iterate_policy_folders([level], dataset=dataset_name))
            sim_results = {name: res for name, res in map(extract_fn, policy_folders)}

            dataset_results[dataset_name] = sim_results

        agg_results = aggregate_for_collect_level(dataset_results)

        for policy_name, error in agg_results.items():
            level_results[policy_name][level] = error

    plot(level_results, levels=args.levels, output_file=args.output_file)

