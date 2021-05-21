import matplotlib.pyplot as plt
import os
import numpy as np
from argparse import ArgumentParser
from functools import partial
from scipy import stats
from collections import namedtuple, OrderedDict
from typing import Any, Dict, List, Optional

from adaptiveleak.utils.constants import POLICIES
from adaptiveleak.utils.file_utils import read_json_gz, iterate_dir
from adaptiveleak.analysis.plot_utils import COLORS, to_label, geometric_mean, MARKER, MARKER_SIZE, LINE_WIDTH, PLOT_STYLE
from adaptiveleak.analysis.plot_utils import PLOT_SIZE, AXIS_FONT, LEGEND_FONT, TITLE_FONT
from adaptiveleak.analysis.plot_utils import extract_results, iterate_policy_folders, dataset_label


WIDTH = 1.0

def plot(sim_results: Dict[str, Dict[float, float]], dataset_name: str, output_file: Optional[str], targets: List[float]):

    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(figsize=PLOT_SIZE, nrows=1, ncols=len(targets))

        if not isinstance(axes, list):
            axes = [axes]

        for i, target in enumerate(sorted(targets)):

            xs: List[int] = []
            ys: List[float] = []

            total_count = sum(sim_results[target].values())

            for bit_width, count in sorted(sim_results[target].items()):
                xs.append(int(bit_width))
                ys.append(float(count) / total_count)

            xs, ys = zip(*sorted(zip(xs, ys)))
            axes[i].bar(xs, ys, width=WIDTH)

            axes[i].set_title('Bit Widths for Target {0:.2f} on {1} Dataset'.format(target, dataset_label(dataset_name)), size=TITLE_FONT)
            axes[i].set_xlabel('Bit Width', size=AXIS_FONT)

            if i == 0:
                axes[i].set_ylabel('Fraction of Sequences', size=AXIS_FONT)

            print('Fraction in bottom two: {0}'.format(ys[0] + ys[1]))


        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file, bbox_inches='tight')
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--targets', type=float, nargs='+', required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    name, sim_results = extract_results(args.folder, field='widths', aggregate_mode=None)

    tokens = list(filter(lambda t: len(t) > 0, args.folder.split(os.sep)))
    dataset = tokens[-3]
    print(dataset)

    plot(sim_results, output_file=args.output_file, dataset_name=dataset, targets=args.targets)
