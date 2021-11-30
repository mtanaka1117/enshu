import numpy as np
import os.path
import math
from argparse import ArgumentParser
from collections import namedtuple
from typing import Dict, List

from adaptiveleak.policies import BudgetWrappedPolicy
from adaptiveleak.utils.file_utils import read_json_gz, iterate_dir, read_json
from adaptiveleak.utils.loading import load_data


Summary = namedtuple('Summary', ['avg', 'std'])


def get_random_sequence(mean: np.ndarray, std: np.ndarray, seq_length: int, rand: np.random.RandomState) -> np.ndarray:
    rand_list: List[np.ndarray] = []

    for m, s in zip(mean, std):
        val = rand.normal(loc=m, scale=s, size=seq_length)  # [T]
        rand_list.append(np.expand_dims(val, axis=-1))

    return np.concatenate(rand_list, axis=-1)  # [T, D]


class HardwareEnergyResult:

    def __init__(self, policy_name: str, collection_rate: int, total_energy: List[float], errors: List[List[int]], num_seq: int):
        # Split the policy name into policy and encoding
        tokens = policy_name.split('_')
        self._policy_name = '_'.join(tokens[:-1])
        self._encoding = tokens[-1]

        self._collection_rate = collection_rate
        self._total_energy = total_energy
        self._errors = errors
        self._num_seq = num_seq

    def get_avg_energy(self) -> float:
        return float(np.average(self._total_energy))

    def get_std_energy(self) -> float:
        return float(np.std(self._total_energy))

    def get_avg_energy_per_seq(self) -> float:
        return float(np.average([(e / self._num_seq) for e in self._total_energy]))

    def get_std_energy_per_seq(self) -> float:
        return float(np.std([(e / self._num_seq) for e in self._total_energy]))

    def get_avg_error(self) -> float:
        return float(np.average(self._errors))

    def get_std_error(self) -> float:
        avg_errors = [np.average(errors) for errors in self._errors]
        return np.std(avg_errors)

    def get_error_for_budget(self, budget: float, inputs: np.ndarray, mean: np.ndarray, std: np.ndarray, rand: np.random.RandomState) -> Summary:
        adjusted_errors: List[float] = []
        avg_errors: List[float] = []

        for trial, (total_energy, errors) in enumerate(zip(self._total_energy, self._errors)):
            if total_energy <= budget:
                adjusted_errors.extend(errors)
                avg_errors.append(np.average(errors))
            else:
                energy_per_seq = total_energy / self._num_seq
                num_collected = int(math.floor(budget / energy_per_seq))

                # Include the errors of samples we can collect under the budget
                adjusted_errors.extend(errors[:num_collected])

                error_sum = sum(errors[:num_collected])

                # For the remaining elements, use random guessing
                offset = trial * self._num_seq
                for idx in range(num_collected, self._num_seq):
                    seq_idx = idx + offset

                    reconstructed = get_random_sequence(mean=mean,
                                                        std=std,
                                                        seq_length=inputs.shape[1],
                                                        rand=rand)

                    error = np.average(np.abs(inputs[seq_idx] - reconstructed))

                    adjusted_errors.append(error)
                    error_sum += error

                avg_errors.append(error_sum / self._num_seq)

        return Summary(avg=np.average(adjusted_errors), std=np.std(avg_errors))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Name of the folder in `devices/results` to analyze.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset.')
    args = parser.parse_args()

    base = os.path.join('..', 'device', 'results', args.folder)

    # Read the data and distribution
    inputs, _ = load_data(dataset_name=args.dataset, fold='mcu')

    distribution_path = os.path.join('..', 'datasets', args.dataset, 'distribution.json')
    distribution = read_json(distribution_path)

    results: Dict[str, Dict[int, HardwareEnergyResult]] = dict()

    for policy_folder in iterate_dir(base, pattern='.*'):
        policy_name = os.path.split(policy_folder)[-1]

        results[policy_name] = dict()

        for budget_folder in iterate_dir(policy_folder, pattern='.*'):

            # Read the energy summary
            energy_path = os.path.join(budget_folder, 'energy.json')
            if not os.path.exists(energy_path):
                continue

            energy_summary = read_json(energy_path)
            num_trials = len(energy_summary['baseline_power'])

            # Get the collection rate
            collection_rate = int(os.path.split(budget_folder)[-1])

            # Read the error logs
            errors: List[List[float]] = []

            for trial in range(num_trials):
                error_log_path = os.path.join(budget_folder, '{0}_{1}_trial{2}.json.gz'.format(policy_name, collection_rate, trial))

                if ('padded' in policy_name) and (not os.path.exists(error_log_path)):
                    policy_file_name = policy_name.replace('padded', 'standard')
                    error_log_path = os.path.join(budget_folder, '{0}_{1}_trial{2}.json.gz'.format(policy_file_name, collection_rate, trial))

                error_log = read_json_gz(error_log_path)
                maes = error_log['maes']

                errors.append(maes)

            results[policy_name][collection_rate] = HardwareEnergyResult(policy_name=policy_name,
                                                                         collection_rate=collection_rate,
                                                                         total_energy=energy_summary['op_energy'],
                                                                         num_seq=energy_summary['num_seq'],
                                                                         errors=errors)

    rand = np.random.RandomState(seed=329)

    # Get the avg energy values (adjusted for the budget)
    for policy_name, policy_results in results.items():
        print('Policy Name: {0}'.format(policy_name))

        for collection_rate, hardware_result in sorted(policy_results.items()):
            if policy_name == 'uniform_standard':
                error = Summary(avg=hardware_result.get_avg_error(), std=hardware_result.get_std_error())
            else:
                uniform_energy = results['uniform_standard'][collection_rate].get_avg_energy()

                error = hardware_result.get_error_for_budget(budget=uniform_energy,
                                                             mean=distribution['mean'],
                                                             std=distribution['std'],
                                                             inputs=inputs,
                                                             rand=rand)

            energy = Summary(avg=hardware_result.get_avg_energy_per_seq(), std=hardware_result.get_std_energy_per_seq())

            print('{0} & {1:.4f} (\\pm {2:.4f}) & {3:.4f} (\\pm {4:.4f})'.format(collection_rate, error.avg, error.std, energy.avg, energy.std))
