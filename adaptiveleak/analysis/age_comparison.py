import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from typing import Dict, DefaultDict, List

from adaptiveleak.analysis.plot_utils import extract_results, iterate_policy_folders


def get_perc_error_results(folder: str, dataset: str) -> DefaultDict[str, List[float]]:
    extract_fn = partial(extract_results, field='mae', aggregate_mode='avg')
    policy_folders = list(iterate_policy_folders([folder], dataset=dataset))

    sim_results = { name: error for name, error in map(extract_fn, policy_folders) }

    # Run the comparison for specific policies and encoding strategies
    policies = ['adaptive_heuristic', 'adaptive_deviation']
    encodings = ['group', 'single_group', 'group_unshifted', 'pruned']

    results: DefaultDict[str, List[float]] = defaultdict(list)

    for policy in policies:
        for encoding in encodings:
            policy_name = '{0}_{1}'.format(policy, encoding)
            base_name = '{0}_group'.format(policy)

            for rate in sim_results[policy_name].keys():
                policy_error = sim_results[policy_name][rate]
                base_error = sim_results[base_name][rate]
                avg_error = (policy_error + base_error) / 2.0

                perc_error = 100.0 * ((policy_error - base_error) / avg_error)

                results[policy_name].append(perc_error)

    return results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Name of the experiment log folder.')
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='A list of dataset names.')
    args = parser.parse_args()

    error_comparison: DefaultDict[str, List[float]] = defaultdict(list)

    print('Number of Datasets: {0}'.format(len(args.datasets)))
    print('==========')

    for dataset in args.datasets:
        dataset_results = get_perc_error_results(args.folder, dataset=dataset)

        for policy, error in dataset_results.items():
            error_comparison[policy].extend(error)

    for policy, normalized_errors in error_comparison.items():
        print('{0} & {1:.4f}%'.format(policy, np.median(normalized_errors)))
