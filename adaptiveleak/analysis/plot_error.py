import matplotlib.pyplot as plt
import os
import numpy as np
from argparse import ArgumentParser
from functools import partial
from collections import namedtuple, OrderedDict
from typing import Any, Dict, List, Optional

from adaptiveleak.utils.constants import POLICIES, ENCODING
from adaptiveleak.utils.file_utils import read_json_gz
from adaptiveleak.analysis.plot_utils import COLORS, to_label, geometric_mean, MARKER, MARKER_SIZE, LINE_WIDTH, PLOT_STYLE
from adaptiveleak.analysis.plot_utils import PLOT_SIZE, AXIS_FONT, LEGEND_FONT, TITLE_FONT
from adaptiveleak.analysis.plot_utils import extract_results, iterate_policy_folders, dataset_label, get_multiplier


def plot(sim_results: Dict[str, Dict[float, float]], dataset_name: str, output_file: Optional[str], is_group_comp: bool, metric: str, include_skip_rnn: bool):

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(PLOT_SIZE[0], PLOT_SIZE[1] * 0.75))

        labels: List[str] = []
        agg_errors: List[float] = []

        policy_names = ['adaptive_heuristic', 'adaptive_deviation'] if is_group_comp else POLICIES

        if not include_skip_rnn:
            policy_names = list(filter(lambda t: not t.startswith('skip_rnn'), policy_names))

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

                if name != 'random' and encoding != 'padded':
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
        ax.set_title('Sampling Error on the {0} Dataset'.format(dataset_label(dataset_name)), fontsize=TITLE_FONT)

        ax.legend(fontsize=LEGEND_FONT)

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file, bbox_inches='tight', transparent=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='The name of the folder holding the experiment logs.')
    parser.add_argument('--dataset', type=str, required=True, help='The dataset name.')
    parser.add_argument('--metric', type=str, choices=['mae', 'rmse'], required=True, help='The error metric to use.')
    parser.add_argument('--output-file', type=str, help='An optional output file in which to save the plot.')
    parser.add_argument('--is-group-comp', action='store_true', help='Whether to use variants of AGE [default: no].')
    parser.add_argument('--include-skip-rnn', action='store_true', help='Whether to include Skip RNNs [default: no)].')
    args = parser.parse_args()

    extract_fn = partial(extract_results, field=args.metric, aggregate_mode=None)
    policy_folders = list(iterate_policy_folders([args.folder], dataset=args.dataset))

    sim_results = {name: res for name, res in map(extract_fn, policy_folders)}
    plot(sim_results, output_file=args.output_file, dataset_name=args.dataset, metric=args.metric, is_group_comp=args.is_group_comp, include_skip_rnn=args.include_skip_rnn)
