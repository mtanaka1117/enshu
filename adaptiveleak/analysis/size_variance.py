import os
import numpy as np
from argparse import ArgumentParser
from functools import partial
from typing import Dict, Any, List

from adaptiveleak.utils.constants import POLICIES
from adaptiveleak.utils.file_utils import read_json_gz
from adaptiveleak.analysis.plot_utils import COLORS, to_label, geometric_mean, MARKER, MARKER_SIZE, LINE_WIDTH, PLOT_STYLE
from adaptiveleak.analysis.plot_utils import PLOT_SIZE, AXIS_FONT, LEGEND_FONT, TITLE_FONT
from adaptiveleak.analysis.plot_utils import extract_results, iterate_policy_folders




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dates', type=str, nargs='+', required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    extract_fn = partial(extract_results, field='num_bytes', aggregate_mode=None)
    policy_folders = iterate_policy_folders(args.dates, dataset=args.dataset)

    sim_results = {name: res for name, res in map(extract_fn, policy_folders)}

    for name, results in sim_results.items():

        std_devs: List[float] = []
        for target, sizes in results.items():
            size_std = np.std(sizes)
            std_devs.append(size_std)
        
        print('{0} & {1:.4f}'.format(name, np.average(std_devs)))

