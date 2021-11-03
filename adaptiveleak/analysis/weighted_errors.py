import os
import numpy as np
from argparse import ArgumentParser
from functools import partial
from typing import Dict, Any, List

from adaptiveleak.utils.constants import POLICIES, SMALL_NUMBER
from adaptiveleak.utils.file_utils import read_json_gz
from adaptiveleak.analysis.plot_utils import COLORS, to_label, geometric_mean, MARKER, MARKER_SIZE, LINE_WIDTH, PLOT_STYLE
from adaptiveleak.analysis.plot_utils import PLOT_SIZE, AXIS_FONT, LEGEND_FONT, TITLE_FONT
from adaptiveleak.analysis.plot_utils import extract_results, iterate_policy_folders



def compute_weighted_errors(error_results: Dict[str, Dict[float, Any]], count_results: Dict[str, Dict[float, Any]]) -> Dict[str, float]:
    for policy_name in ['adaptive_heuristic_group', 'adaptive_deviation_group', 'uniform_standard', 'adaptive_heuristic_standard', 'adaptive_deviation_standard']:

        errors_list: List[float] = []
        for budget in sorted(error_results[policy_name].keys()):

            weighted_errors = 0.0
            total_weight = 0.0

            for error, measurement_count in zip(error_results[policy_name][budget], count_results[policy_name][budget]):
                weighted_errors += measurement_count * error
                total_weight += measurement_count

            weighted_error = weighted_errors / total_weight
            errors_list.append(weighted_error)

        print('{} & {}'.format(policy_name, np.average(errors_list)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dates', type=str, nargs='+', required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    policy_folders = list(iterate_policy_folders(args.dates, dataset=args.dataset))

    error_fn = partial(extract_results, field='all_mae', aggregate_mode=None)
    error_results = {name: res for name, res in map(error_fn, policy_folders)}

    count_fn = partial(extract_results, field='num_measurements', aggregate_mode=None)
    count_results = {name: res for name, res in map(count_fn, policy_folders)}

    compute_weighted_errors(error_results=error_results, count_results=count_results)    
