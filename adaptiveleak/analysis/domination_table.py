import matplotlib.pyplot as plt
import os
import numpy as np
from argparse import ArgumentParser
from functools import partial
from scipy import stats
from collections import namedtuple, OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, DefaultDict, Iterable, Tuple

from adaptiveleak.utils.constants import POLICIES, ENCODING, SMALL_NUMBER
from adaptiveleak.utils.file_utils import read_json_gz, iterate_dir
from adaptiveleak.analysis.plot_utils import COLORS, to_label, geometric_mean, MARKER, MARKER_SIZE, LINE_WIDTH, PLOT_STYLE
from adaptiveleak.analysis.plot_utils import PLOT_SIZE, AXIS_FONT, LEGEND_FONT, TITLE_FONT
from adaptiveleak.analysis.plot_utils import extract_results, iterate_policy_folders, dataset_label


def get_policy_names(use_shifted: bool) -> Iterable[str]:
    adaptive_encodings = ['group', 'group_unshifted', 'single_group', 'pruned'] if use_shifted else ['standard', 'padded', 'group']
    policies = ['adaptive_heuristic', 'adaptive_deviation'] if use_shifted else POLICIES

    for name in policies:
        encodings = adaptive_encodings if name not in ('uniform', 'random') else ['standard']

        for encoding in encodings:
            policy_name = '{0}_{1}'.format(name, encoding)

            yield policy_name


def make_comparisons(base_policy: str, dataset_results: DefaultDict[str, Dict[str, List[float]]], use_shifted: bool) -> List[Tuple[int, int, int]]:
    results: List[Tuple[int, int, int]] = []

    for policy_name in get_policy_names(use_shifted=use_shifted):
        first_wins = 0
        second_wins = 0
        ties = 0

        for dataset in dataset_results.keys():
            # Skip unknown policies
            if (policy_name not in dataset_results[dataset]) or (base_policy not in dataset_results[dataset]):
                continue

            for first_mae, second_mae in zip(dataset_results[dataset][base_policy], dataset_results[dataset][policy_name]):
                if abs(first_mae - second_mae) < SMALL_NUMBER:
                    ties += 1
                elif (first_mae < second_mae):
                    first_wins += 1
                else:
                    second_wins += 1
                    
        results.append((first_wins, second_wins, ties))

    return results


def make_table(dataset_results: Dict[str, Dict[str, List[float]]], use_shifted: bool):
    labels: List[str] = []
    agg_errors: List[float] = []

    policy_names: List[str] = list(get_policy_names(use_shifted=use_shifted))

    print(' & ' + ' & '.join(policy_names))

    for policy_name in policy_names:
        comparisons = make_comparisons(base_policy=policy_name,
                                        dataset_results=dataset_results,
                                        use_shifted=use_shifted)

        comparison_fmt = ['{0}/{1}/{2}'.format(f, s, t) for f, s, t in comparisons]

        print('{0} & {1} \\\\'.format(policy_name, ' & '.join(comparison_fmt)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--use-shifted', action='store_true')
    args = parser.parse_args()

    dataset_results: DefaultDict[str, Dict[str, List[float]]] = defaultdict(dict)  # Map of policy -> { data-set -> [MAE] }
    base = os.path.join('..', 'saved_models')

    extract_fn = partial(extract_results, field='mae', aggregate_mode=None)

    print('Num Datasets: {0}'.format(len(args.datasets)))
    print('========')

    for dataset in args.datasets:
        policy_folders = list(iterate_dir(os.path.join(base, dataset, args.folder), pattern='.*'))

        for name, res in map(extract_fn, policy_folders):
            if len(name) == 0:
                continue

            dataset_results[dataset][name] = [error for _, error in sorted(res.items())]

    make_table(dataset_results, use_shifted=args.use_shifted)

