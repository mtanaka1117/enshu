import csv
import h5py
import os.path
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from typing import Dict, List

from adaptiveleak.policies import run_policy, BudgetWrappedPolicy
from adaptiveleak.energy_systems import BluetoothEnergy
from adaptiveleak.utils.file_utils import read_json, iterate_dir
from adaptiveleak.utils.constants import PERIOD
from adaptiveleak.utils.types import EncryptionMode


WIDTH = 0.25


def get_energy(path: str) -> float:
    with open(path, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')

        for idx, line in enumerate(reader):
            if idx == 0:
                continue

            energy = float(line[-1])

    return energy / 1000.0  # Return the energy in mJ


def simulate_policy(policy_name: str, inputs: np.ndarray, collection_rate: float, dataset_name: str) -> float:
    """
    Runs the policy on the given inputs and returns the total energy.
    """
    tokens = policy_name.split('_')
    policy_name = '_'.join(tokens[0:-1])
    encoding_mode = tokens[-1]

    policy = BudgetWrappedPolicy(name=policy_name,
                                 seq_length=inputs.shape[1],
                                 num_features=inputs.shape[2],
                                 dataset=dataset_name,
                                 collection_rate=collection_rate,
                                 encryption_mode=EncryptionMode.STREAM,
                                 encoding=encoding_mode,
                                 should_compress=False)

    policy.init_for_experiment(num_sequences=len(inputs) * 2)

    energy_list: List[float] = []
    for idx, sequence in enumerate(inputs):
        # Run the policy on the given sequences
        policy.reset()
        policy_result = run_policy(policy=policy, sequence=sequence)
        policy.step(seq_idx=idx, count=policy_result.num_collected)
        
        # Record the energy
        energy_list.append(policy_result.energy)

    return np.sum(energy_list)


def plot(simulated_energy: Dict[str, float], trace_energy: Dict[str, float]):

    with plt.style.context('seaborn-ticks'):
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(12, 8))

        normalized_name = next(iter(sorted(simulated_energy.keys())))

        # Group the energy values between simulated and trace
        grouped: Dict[str, List[float]] = dict()
        normalized: Dict[str, List[float]] = dict()

        for name in sorted(simulated_energy.keys()):
            grouped[name] = [simulated_energy[name], trace_energy[name]]
            normalized[name] = [simulated_energy[name] / simulated_energy[normalized_name], trace_energy[name] / trace_energy[normalized_name]]

        xs = np.arange(0, 2)

        for idx, name in enumerate(sorted(simulated_energy.keys())):
            x_values = (idx - 1) * WIDTH + xs

            ax1.bar(x_values, grouped[name], width=WIDTH, label=name)
            ax2.bar(x_values, normalized[name], width=WIDTH, label=name)

            for i in xs:
                ax1.annotate('{0:.3f}'.format(grouped[name][i]),
                             xy=(x_values[i], grouped[name][i]),
                             xytext=(x_values[i] - 0.1, grouped[name][i] * 1.02))

                ax2.annotate('{0:.3f}'.format(normalized[name][i]),
                             xy=(x_values[i], normalized[name][i]),
                             xytext=(x_values[i] - 0.1, normalized[name][i] * 1.02))

        #ax1.legend()
        ax2.legend(framealpha=1.0, facecolor='w', loc='lower left', frameon=True)

        ax1.set_xticks(xs)
        ax1.set_xticklabels(['Simulated', 'Actual'])

        ax2.set_xticks(xs)
        ax2.set_xticklabels(['Simulated', 'Actual'])

        ax1.set_ylabel('Energy (mJ) / Sequence')
        ax2.set_ylabel('Normalized Energy / Sequence')

        ax1.set_title('Avg Energy per Seq over 5 Trials')
        ax2.set_title('Normalized Avg Energy per Seq over 5 Trials')

        plt.savefig('activity_energy_5_seq.pdf')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--policies', type=str, nargs='+', required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--collection-rate', type=float, required=True)
    args = parser.parse_args()

    base = os.path.join('..', 'traces', 'end_to_end', args.dataset)

    # Read the experimental parameters
    setup = read_json(os.path.join(base, 'setup.json'))
    num_seq = setup['num_sequences']
    num_seconds = setup['time']

    # Calculate the baseline Bluetooth Energy
    baseline_folder = os.path.join('..', 'traces', 'baseline')

    baseline_list: List[float] = []
    for path in iterate_dir(baseline_folder, pattern='.*csv'):
        baseline_list.append(get_energy(path=path))

    # The standard reading occurs over 10 seconds, so we scale it to the number of seconds in the experiment
    baseline_energy = np.average(baseline_list) * (num_seconds / 10)
    
    # Adjust the baseline energy to be the portion in which the system is idle
    used_seconds = PERIOD * num_seq
    idle_seconds = num_seconds - used_seconds
    excess_energy = baseline_energy * (idle_seconds / num_seconds)

    # Add the energy required to send the start byte
    bt_energy = BluetoothEnergy()
    excess_energy += bt_energy.get_energy(num_bytes=1, use_noise=False)

    # Get the trace results
    trace_energy: Dict[str, float] = dict()
    for policy_name in args.policies:
        path = os.path.join(base, '{0}_{1}.csv'.format(policy_name, int(round(args.collection_rate * 100))))
        trace_energy[policy_name] = get_energy(path=path) - excess_energy

    # Read the data
    with h5py.File(os.path.join('..', 'datasets', args.dataset, 'mcu', 'data.h5'), 'r') as fin:
        inputs = fin['inputs'][:num_seq]

    # Get the simulated results
    simulated_energy: Dict[str, float] = dict()
    for policy_name in args.policies:
        simulated_energy[policy_name] = simulate_policy(policy_name=policy_name,
                                                        inputs=inputs,
                                                        collection_rate=args.collection_rate,
                                                        dataset_name=args.dataset)


    # Plot the results
    plot(simulated_energy=simulated_energy,
         trace_energy=trace_energy)
