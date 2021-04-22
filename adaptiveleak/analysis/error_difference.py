import os
import numpy as np
from argparse import ArgumentParser
from functools import partial
from typing import Dict, Any

from adaptiveleak.utils.constants import POLICIES
from adaptiveleak.utils.file_utils import read_json_gz
from adaptiveleak.analysis.plot_utils import COLORS, to_label, geometric_mean, MARKER, MARKER_SIZE, LINE_WIDTH, PLOT_STYLE
from adaptiveleak.analysis.plot_utils import PLOT_SIZE, AXIS_FONT, LEGEND_FONT, TITLE_FONT
from adaptiveleak.analysis.plot_utils import extract_results, iterate_policy_folders



def compute_average_loss(sim_results: Dict[str, Dict[float, Any]]) -> float:
    total_perc_diff = 0.0
    total_count = 0.0

    for policy_name in POLICIES:
        if (not policy_name.endswith('group')) or (policy_name not in sim_results):
            continue

        standard_name = policy_name.replace('group', 'standard')

        policy_error = np.average(list(sim_results[policy_name].values()))
        standard_error = np.average(list(sim_results[standard_name].values()))

        middle = (policy_error + standard_error) / 2.0
        avg_perc_diff = (policy_error - standard_error) / middle * 100.0

        total_perc_diff += avg_perc_diff
        total_count += 1.0

    print('Retrieved {0} Policies'.format(int(total_count)))
    return total_perc_diff / total_count


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dates', type=str, nargs='+', required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    extract_fn = partial(extract_results, field='errors', aggregate_mode='avg')
    policy_folders = iterate_policy_folders(args.dates, dataset=args.dataset)

    sim_results = {name: res for name, res in map(extract_fn, policy_folders)}
    print('Average Symmetric Percentage Difference: {0:.2f}%'.format(compute_average_loss(sim_results)))
