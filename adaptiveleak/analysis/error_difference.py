import os
import numpy as np
from argparse import ArgumentParser
from functools import partial
from typing import Dict, Any

from adaptiveleak.utils.constants import POLICIES, SMALL_NUMBER
from adaptiveleak.utils.file_utils import read_json_gz
from adaptiveleak.analysis.plot_utils import COLORS, to_label, geometric_mean, MARKER, MARKER_SIZE, LINE_WIDTH, PLOT_STYLE
from adaptiveleak.analysis.plot_utils import PLOT_SIZE, AXIS_FONT, LEGEND_FONT, TITLE_FONT
from adaptiveleak.analysis.plot_utils import extract_results, iterate_policy_folders



def compute_max_differences(sim_results: Dict[str, Dict[float, Any]]) -> Dict[str, float]:
    results: Dict[str, float] = dict()

    for policy_name in ['adaptive_heuristic_group', 'adaptive_deviation_group']:
        standard_name = policy_name.replace('group', 'standard')

        max_diff = 0.0
        max_perc = 0.0
        for budget in sim_results[policy_name].keys():
            for p, s in zip(sim_results[policy_name][budget], sim_results[standard_name][budget]):
                diff = p - s
                perc_diff = diff / (s + SMALL_NUMBER)

                if diff > max_diff:
                    max_diff = diff
                    max_perc = perc_diff
                    print('p: {:.4f}, s: {:.4f}'.format(p, s))

        results[policy_name] = (max_diff, max_perc * 100.0)

    return results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dates', type=str, nargs='+', required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    extract_fn = partial(extract_results, field='all_mae', aggregate_mode=None)
    policy_folders = iterate_policy_folders(args.dates, dataset=args.dataset)

    sim_results = {name: res for name, res in map(extract_fn, policy_folders)}
    results = compute_max_differences(sim_results)
    print(results)
