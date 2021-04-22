import matplotlib.pyplot as plt
import os
import numpy as np
from argparse import ArgumentParser
from functools import partial
from scipy import stats
from collections import namedtuple, OrderedDict
from typing import Any, Dict, List, Optional

from adaptiveleak.utils.constants import POLICIES
from adaptiveleak.utils.file_utils import read_json_gz
from adaptiveleak.analysis.plot_utils import COLORS, to_label, geometric_mean, MARKER, MARKER_SIZE, LINE_WIDTH, PLOT_STYLE
from adaptiveleak.analysis.plot_utils import PLOT_SIZE, AXIS_FONT, LEGEND_FONT, TITLE_FONT
from adaptiveleak.analysis.plot_utils import extract_results, iterate_policy_folders



def plot(size_results: Dict[str, Dict[float, float]], count_results: Dict[str, Dict[float, float]], dataset_name: str, output_file: Optional[str]):

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=PLOT_SIZE)

        for name in POLICIES:
            if name not in size_results:
                continue

            targets = list(sorted(size_results[name].keys()))
            sizes = [size_results[name][t] for t in targets]
            counts = [count_results[name][t] for t in targets]

            ax.plot(targets, sizes, marker=MARKER, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=to_label(name), color=COLORS[name])

            print('{0} & {1:.2f} ({2:.2f})'.format(name, np.average(sizes), np.average(counts)))

        ax.set_xlabel('Fraction of Measurements', fontsize=AXIS_FONT)
        ax.set_ylabel('Avg Message Size (bytes)', fontsize=AXIS_FONT)
        ax.set_title('Average Message Size on the {0} Dataset'.format(dataset_name.capitalize()), fontsize=TITLE_FONT)

        ax.legend(fontsize=LEGEND_FONT)

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file, bbox_inches='tight')
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dates', type=str, nargs='+', required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    size_fn = partial(extract_results, field='num_bytes', aggregate_mode='avg')
    count_fn = partial(extract_results, field='num_measurements', aggregate_mode='avg')

    policy_folders = list(iterate_policy_folders(args.dates, dataset=args.dataset))

    size_results = {name: res for name, res in map(size_fn, policy_folders)}
    count_results = {name: res for name, res in map(count_fn, policy_folders)}

    plot(size_results, count_results, output_file=args.output_file, dataset_name=args.dataset)

