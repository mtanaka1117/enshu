import matplotlib.pyplot as plt
import os
import numpy as np
from argparse import ArgumentParser
from functools import partial
from scipy import stats
from collections import namedtuple, OrderedDict
from typing import Any, Dict, List, Optional

from adaptiveleak.utils.constants import POLICIES, ENCODING
from adaptiveleak.utils.file_utils import read_json_gz
from adaptiveleak.analysis.plot_utils import COLORS, to_label, geometric_mean, MARKER, MARKER_SIZE, LINE_WIDTH, PLOT_STYLE
from adaptiveleak.analysis.plot_utils import PLOT_SIZE, AXIS_FONT, LEGEND_FONT, TITLE_FONT
from adaptiveleak.analysis.plot_utils import extract_results, iterate_policy_folders, dataset_label


def plot(sim_results: Dict[str, Dict[float, float]], dataset_name: str, output_file: Optional[str], is_group_comp: bool, metric: str):

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=PLOT_SIZE)

        labels: List[str] = []
        agg_errors: List[float] = []

        policy_names = ['adaptive_heuristic', 'adaptive_deviation'] if is_group_comp else POLICIES
        encoding_names = ['single_group', 'group_unshifted', 'pruned', 'group'] if is_group_comp else ['standard', 'padded', 'group']

        for name in policy_names:
            encodings = encoding_names if name not in ('uniform', 'random') else ['standard']

            for encoding in encodings:

                policy_name = '{0}_{1}'.format(name, encoding)

                if (policy_name not in sim_results) and (name not in sim_results):
                    continue

                if name in sim_results:
                    policy_name = name

                model_results = sim_results[policy_name]
                energy_per_seq = list(sorted(model_results.keys()))
                errors = [model_results[e] for e in energy_per_seq]

                print('Num Budgets: {0}'.format(len(errors)))

                ax.plot(energy_per_seq, errors, marker=MARKER, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=to_label(policy_name), color=COLORS[policy_name])

                avg = np.average(errors)
                if metric in ('norm_mae', 'norm_rmse'):
                    avg = avg * 100

                agg_errors.append(avg)

                if encoding == 'standard':
                    labels.append(name)
                else:
                    labels.append(policy_name)

        min_error = np.argmin(agg_errors)

        print(' & '.join(labels))
        print(' & '.join((('{0:.5f}'.format(x) if i != min_error else '\\textbf{{{0:.5f}}}'.format(x)) for i, x in enumerate(agg_errors))))

        ax.set_xlabel('Energy Budget (Average mJ / Seq)', fontsize=AXIS_FONT)
        ax.set_ylabel(metric.upper(), fontsize=AXIS_FONT)
        ax.set_title('Average Reconstruction Error on the {0} Dataset'.format(dataset_label(dataset_name)), fontsize=TITLE_FONT)

        ax.legend(fontsize=LEGEND_FONT)

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file, bbox_inches='tight', transparent=True)
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dates', type=str, nargs='+', required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--metric', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--is-group-comp', action='store_true')
    args = parser.parse_args()

    extract_fn = partial(extract_results, field=args.metric, aggregate_mode=None)
    policy_folders = list(iterate_policy_folders(args.dates, dataset=args.dataset))

    sim_results = {name: res for name, res in map(extract_fn, policy_folders)}
    plot(sim_results, output_file=args.output_file, dataset_name=args.dataset, metric=args.metric, is_group_comp=args.is_group_comp)
