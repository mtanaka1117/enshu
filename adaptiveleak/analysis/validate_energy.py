import csv
import h5py
import os.path
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from collections import namedtuple
from typing import Dict, List

from adaptiveleak.policies import run_policy, BudgetWrappedPolicy
from adaptiveleak.energy_systems import BluetoothEnergy
from adaptiveleak.utils.file_utils import read_json, iterate_dir
from adaptiveleak.utils.constants import PERIOD
from adaptiveleak.utils.types import EncryptionMode
from adaptiveleak.analysis.plot_utils import COLORS


WIDTH = 0.15
EnergyResult = namedtuple('EnergyResult', ['comm', 'total', 'comp'])


def get_e2e_energy(folder: str, num_seq: int) -> EnergyResult:
    energy_dict = read_json(os.path.join(folder, 'energy.json'))

    total_list: List[float] = [x / num_seq for x in energy_dict['op_energy']]
    comm_list: List[float] = [x / num_seq for x in energy_dict['comm_energy']]
    comp_list: List[float] = [x / num_seq for x in energy_dict['comp_energy']]

    return EnergyResult(comm=np.median(comm_list),
                        total=np.median(total_list),
                        comp=np.median(comp_list))


def simulate_policy(policy_name: str, inputs: np.ndarray, collection_rate: float, dataset_name: str) -> EnergyResult:
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

    # Track the energy required for communication
    bt_energy = BluetoothEnergy()

    total_energy_list: List[float] = []
    comm_energy_list: List[float] = []
    comp_energy_list: List[float] = []

    for idx, sequence in enumerate(inputs):
        # Run the policy on the given sequences
        policy.reset()
        policy_result = run_policy(policy=policy, sequence=sequence, should_enforce_budget=False)
        policy.step(seq_idx=idx, count=policy_result.num_collected)

        # Record the communication energy
        #comm_energy = bt_energy.get_energy(num_bytes=policy_result.num_bytes,
        #                                   use_noise=False)
        comm_energy = policy.energy_unit.get_communication_energy(num_bytes=policy_result.num_bytes,
                                                                  use_noise=False)
        comm_energy_list.append(comm_energy)

        # Record the computation energy
        comp_energy = policy.energy_unit.get_computation_energy(num_bytes=policy_result.num_bytes,
                                                                num_collected=policy_result.num_collected,
                                                                use_noise=False)
        comp_energy_list.append(comp_energy)

        # Record the total energy
        total_energy_list.append(policy_result.energy)

    return EnergyResult(total=np.average(total_energy_list),
                        comm=np.average(comm_energy_list),
                        comp=np.average(comp_energy_list))


def plot(simulated_energy: Dict[str, EnergyResult], trace_energy: Dict[str, EnergyResult]):

    with plt.style.context('seaborn-ticks'):
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(12, 8))

        normalized_name = next(iter(sorted(simulated_energy.keys())))

        # Group the energy values between simulated and trace
        grouped: Dict[str, List[float]] = dict()
        normalized: Dict[str, List[float]] = dict()

        for name in sorted(simulated_energy.keys()):
            grouped[name] = [simulated_energy[name], trace_energy[name]]

        xs = np.arange(0, 4)

        for idx, name in enumerate(sorted(simulated_energy.keys())):
            x_values = (idx - 2) * WIDTH + xs

            raw_upper = []        
            raw_upper.append(simulated_energy[name].total - simulated_energy[name].comm)
            raw_upper.append(trace_energy[name].total - trace_energy[name].comm)
            raw_upper.append(simulated_energy[name].comp)
            raw_upper.append(trace_energy[name].comp)

            raw_lower = []
            raw_lower.append(simulated_energy[name].comm)
            raw_lower.append(trace_energy[name].comm)
            raw_lower.append(0)
            raw_lower.append(0)

            ax1.bar(x_values, raw_lower, width=WIDTH, color=COLORS[name], hatch='/', edgecolor='k')
            ax1.bar(x_values, raw_upper, width=WIDTH, bottom=raw_lower, color=COLORS[name], label=name, edgecolor='k')

            norm_upper = []
            norm_upper.append((simulated_energy[name].total - simulated_energy[name].comm) / simulated_energy[normalized_name].total)
            norm_upper.append((trace_energy[name].total - trace_energy[name].comm) / trace_energy[normalized_name].total)
            norm_upper.append(simulated_energy[name].comp / simulated_energy[normalized_name].total)
            norm_upper.append(trace_energy[name].comp / trace_energy[normalized_name].total)

            norm_lower = []
            norm_lower.append(simulated_energy[name].comm / simulated_energy[normalized_name].total)
            norm_lower.append(trace_energy[name].comm / trace_energy[normalized_name].total)
            norm_lower.append(0)
            norm_lower.append(0)

            ax2.bar(x_values, norm_lower, width=WIDTH, color=COLORS[name], hatch='/', edgecolor='k')
            ax2.bar(x_values, norm_upper, width=WIDTH, bottom=norm_lower, color=COLORS[name], label=name, edgecolor='k')

            for i in xs:
                if i < 2:
                    raw = raw_upper[i] + raw_lower[i]
                    norm = norm_upper[i] + norm_lower[i]
                else:
                    raw = raw_upper[i]
                    norm = norm_upper[i]

                raw_y = 0.5 if idx % 2 == 0 else 2
                norm_y = 0.01 if idx % 2 == 0 else 0.03

                ax1.annotate('{0:.2f}'.format(raw),
                             xy=(x_values[i], raw),
                             xytext=(x_values[i] - 0.1, raw + raw_y))

                ax2.annotate('{0:.2f}'.format(norm),
                             xy=(x_values[i], norm),
                             xytext=(x_values[i] - 0.1, norm + norm_y))

        #ax1.legend()
        ax2.legend(framealpha=1.0, facecolor='w', loc='lower left', frameon=True)

        ax1.set_xticks(xs)
        ax1.set_xticklabels(['Sim', 'Actual', 'Sim Comp', 'Actual Comp'])

        ax2.set_xticks(xs)
        ax2.set_xticklabels(['Sim', 'Actual', 'Sim Comp', 'Actual Comp'])

        ax1.set_ylabel('Energy (mJ) / Sequence')
        ax2.set_ylabel('Normalized Energy / Sequence')

        ax1.set_title('Avg Energy per Sequence')
        ax2.set_title('Normalized Avg Energy per Sequence')

        plt.savefig('activity_energy_2_seq.png', transparent=True, bbox_inches='tight')
        #plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--policies', type=str, nargs='+', required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    base = os.path.join('..', 'traces', 'end_to_end', args.dataset)

    # Read the experimental parameters
    setup = read_json(os.path.join(base, 'setup.json'))
    num_seq = setup['num_sequences']
    num_seconds = setup['time']
    collection_rate = setup['collection_rate']

    # Get the trace results
    trace_energy: Dict[str, float] = dict()
    for policy_name in args.policies:
        folder = os.path.join(base, policy_name)
        trace_energy[policy_name] = get_e2e_energy(folder=folder, num_seq=num_seq)

    # Read the data
    with h5py.File(os.path.join('..', 'datasets', args.dataset, 'mcu', 'data.h5'), 'r') as fin:
        inputs = fin['inputs'][:num_seq]

    # Get the simulated results
    simulated_energy: Dict[str, float] = dict()
    for policy_name in args.policies:
        simulated_energy[policy_name] = simulate_policy(policy_name=policy_name,
                                                        inputs=inputs,
                                                        collection_rate=collection_rate,
                                                        dataset_name=args.dataset)

    # Plot the results
    plot(simulated_energy=simulated_energy,
         trace_energy=trace_energy)
