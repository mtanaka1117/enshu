import numpy as np
import os.path
import math
from argparse import ArgumentParser
from collections import namedtuple
from typing import Dict, List, Set, Any

from adaptiveleak.policies import BudgetWrappedPolicy
from adaptiveleak.utils.file_utils import read_json_gz, iterate_dir, read_json
from adaptiveleak.utils.loading import load_data


Summary = namedtuple('Summary', ['energy', 'mae', 'num_collected'])


def get_overlapping_indices(policy_results: Dict[str, Dict[int, Summary]], rate: int, max_collected: int) -> Set[int]:
    indices: Set[int] = set(range(2000))  # Large enough for all tasks

    for policy_result in policy_results.values():
        summary = policy_result[rate]

        num_collected = summary.num_collected
        max_collected = len(summary.energy)

        if num_collected < max_collected:
            indices_to_keep = np.argsort(summary.mae)[:num_collected]
        else:
            indices_to_keep = list(range(max_collected))

        index_set = set(indices_to_keep)
        indices = indices.intersection(index_set)

    return indices


def filter_list(values: List[Any], idx_to_keep: Set[int]) -> List[Any]:
    return [x for i, x in enumerate(values) if i in idx_to_keep]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--num-trials', type=int, required=True)
    parser.add_argument('--num-seq', type=int, required=True)
    args = parser.parse_args()

    base = os.path.join('..', 'device', 'results', args.dataset)

    results: Dict[str, Dict[int, Summary]] = dict()

    for policy_folder in iterate_dir(base, pattern='.*'):
        policy_name = os.path.split(policy_folder)[-1]
        results[policy_name] = dict()

        for budget_folder in iterate_dir(policy_folder, pattern='.*'):

            mae_list: List[float] = []
            num_collected = 0

            # Read the energy summary
            energy_path = os.path.join(budget_folder, 'energy.json')
            if not os.path.exists(energy_path):
                continue

            energy_summary = read_json(energy_path)
            energy_list: List[float] = energy_summary['op_energy']

            # Get the collection rate
            try:
                collection_rate = int(os.path.split(budget_folder)[-1])
            except ValueError:
                continue

            # Read the error logs
            mae_list: List[float] = []
            num_collected = 0

            for trial in range(args.num_trials):
                error_log_path = os.path.join(budget_folder, '{0}_{1}_trial{2}.json.gz'.format(policy_name, collection_rate, trial))
                error_log = read_json_gz(error_log_path)
                maes = error_log['maes']

                mae_list.extend(maes)
                num_collected += error_log['recv_count']

            results[policy_name][collection_rate] = Summary(mae=mae_list,
                                                            energy=energy_list,
                                                            num_collected=num_collected)

        # Remove any empty results
        if len(results[policy_name]) == 0:
            del results[policy_name]

    collection_rates = list(sorted(results['uniform_standard'].keys()))

    for rate in collection_rates:
        idx_to_keep = get_overlapping_indices(policy_results=results, rate=rate, max_collected=args.num_seq * args.num_trials)

        print('Collection Rate: {0}'.format(rate))

        for policy_name, policy_result in results.items():
            energy = filter_list(policy_result[rate].energy, idx_to_keep)
            errors = filter_list(policy_result[rate].mae, idx_to_keep)

            med_energy = np.median(energy)
            iqr_energy = np.percentile(energy, 75) - np.percentile(energy, 25)

            avg_error = np.average(errors)
            std_error = np.std(errors)

            print('{0} & {1:.4f} (\\pm {2:.4f}) & {3:.4f} (\\pm {4:.5f})'.format(policy_name, avg_error, std_error, med_energy, iqr_energy)) 

        print()

