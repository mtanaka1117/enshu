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



def percent_errors_for_dataset(date: str, dataset: str, policy_one: str, policy_two: str) -> Dict[str, List[float]]:
    extract_fn = partial(extract_results, field='mae', aggregate_mode=None)
    policy_folders = list(iterate_policy_folders([date], dataset=dataset))

    sim_results = {name: res for name, res in map(extract_fn, policy_folders)}

    policy_one_results = sim_results[policy_one]
    policy_two_results = sim_results[policy_two]

    energy_budgets = list(sorted(policy_one_results.keys()))

    return {
        policy_one: [(policy_one_results[b] - policy_two_results[b]) / ((policy_one_results[b] + policy_two_results[b]) / 2 + SMALL_NUMBER) for b in energy_budgets],
        policy_two: [(policy_two_results[b] - policy_one_results[b]) / ((policy_one_results[b] + policy_two_results[b]) / 2 + SMALL_NUMBER) for b in energy_budgets]
    }
 

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--date', type=str, required=True)
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--policy-one', type=str, required=True)
    parser.add_argument('--policy-two', type=str, required=True)
    args = parser.parse_args()

    print('Num Datasets: {0}'.format(len(args.datasets)))
    print('==========')

    policy_one_errors: List[float] = []
    policy_two_errors: List[float] = []

    for dataset in args.datasets:
        percent_errors = percent_errors_for_dataset(date=args.date, dataset=dataset, policy_one=args.policy_one, policy_two=args.policy_two)

        policy_one_errors.extend(percent_errors[args.policy_one])
        policy_two_errors.extend(percent_errors[args.policy_two])


    print('{0}: {1:.5f}%'.format(args.policy_one, np.median(policy_one_errors) * 100.0))
    print('{0}: {1:.5f}%'.format(args.policy_two, np.median(policy_two_errors) * 100.0))
