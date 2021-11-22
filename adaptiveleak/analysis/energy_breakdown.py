import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import namedtuple
from typing import List

from adaptiveleak.energy_systems import EnergyUnit
from adaptiveleak.utils.constants import PERIOD
from adaptiveleak.utils.data_utils import calculate_bytes
from adaptiveleak.utils.loading import load_data
from adaptiveleak.utils.types import PolicyType, EncryptionMode, EncodingMode, CollectMode


LABELS = ['Collect', 'Should Collect', 'Update', 'Encode', 'Encrypt', 'BLE', 'Total']
EnergyBreakdown = namedtuple('EnergyBreakdown', ['collect', 'should_collect', 'update', 'encode', 'encrypt', 'comm', 'total'])


def get_breakdown(energy_unit: EnergyUnit, seq_length: int, num_features: int, num_collected: int, num_bytes: int):
    # Get the total energy
    total_energy = energy_unit.get_energy(num_collected=num_collected,
                                          num_bytes=num_bytes,
                                          use_noise=False)

    # Get the energy from each component
    collect_energy = energy_unit._collect.get_energy_multiple(count=num_collected,
                                                              use_noise=False)

    should_collect_energy = energy_unit._should_collect.get_energy_multiple(count=(seq_length - num_collected),
                                                                            use_noise=False)  # The update accounts for 'should_collect'

    update_energy = energy_unit._update.get_energy_multiple(count=num_collected,
                                                            use_noise=False)

    encode_energy = energy_unit._encode.get_energy(num_features=num_features * num_collected,
                                                   use_noise=False)

    encrypt_energy = energy_unit._encrypt.get_energy(num_bytes=num_bytes,
                                                     use_noise=False)

    comm_energy = energy_unit._comm.get_energy(num_bytes=num_bytes,
                                               use_noise=False)

    return EnergyBreakdown(collect=collect_energy,
                           should_collect=should_collect_energy,
                           update=update_energy,
                           encode=encode_energy,
                           encrypt=encrypt_energy,
                           comm=comm_energy,
                           total=total_energy)


def print_table(standard: EnergyBreakdown, group: EnergyBreakdown):
    standard_list = breakdown_to_list(standard, normalize=False)
    standard_list_norm = breakdown_to_list(standard, normalize=True)

    group_list = breakdown_to_list(group, normalize=False)
    group_list_norm = breakdown_to_list(group, normalize=True)

    print(' & '.join(LABELS))
    print(' & '.join(['{0:.4f} ({1:.2f}\\%)'.format(s, n) for s, n in zip(standard_list, standard_list_norm)]))
    print(' & '.join(['{0:.4f} ({1:.2f}\\%)'.format(s, n) for s, n in zip(group_list, group_list_norm)]))


def breakdown_to_list(breakdown: EnergyBreakdown, normalize: bool) -> List[float]:
    extracted = [breakdown.collect, breakdown.should_collect, breakdown.update, breakdown.encode, breakdown.encrypt, breakdown.comm, breakdown.total]

    if normalize:
        return [(x / breakdown.total) * 100 for x in extracted]

    return extracted


def plot_breakdowns(standard: EnergyBreakdown, group: EnergyBreakdown):
    with plt.style.context('seaborn-ticks'):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

        ax1.pie(breakdown_to_list(standard), labels=LABELS, autopct='%1.3f%%', startangle=90)
        ax1.axis('equal')

        plt.show()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True, choices=['uniform', 'adaptive_heuristic', 'adaptive_deviation', 'skip_rnn'])
    args = parser.parse_args()

    # Load the data and get dimensions
    inputs, _ = load_data(args.dataset, fold='validation')
    num_seq, seq_length, num_features = inputs.shape

    # Get the target number of bytes
    message_bytes = calculate_bytes(width=16,
                                    num_collected=seq_length,
                                    num_features=num_features,
                                    seq_length=seq_length,
                                    encryption_mode=EncryptionMode.STREAM)

    # Create the energy units
    standard_unit = EnergyUnit(policy_type=PolicyType[args.policy.upper()],
                               encryption_mode=EncryptionMode.STREAM,
                               encoding_mode=EncodingMode.STANDARD,
                               collect_mode=CollectMode.LOW,
                               seq_length=seq_length,
                               num_features=num_features,
                               period=PERIOD)

    group_unit = EnergyUnit(policy_type=PolicyType[args.policy.upper()],
                            encryption_mode=EncryptionMode.STREAM,
                            encoding_mode=EncodingMode.GROUP,
                            collect_mode=CollectMode.LOW,
                            seq_length=seq_length,
                            num_features=num_features,
                            period=PERIOD)

    # Get the energy breakdown for both units
    standard_breakdown = get_breakdown(energy_unit=standard_unit,
                                       num_collected=seq_length,
                                       num_bytes=message_bytes,
                                       seq_length=seq_length,
                                       num_features=num_features)

    group_breakdown = get_breakdown(energy_unit=group_unit,
                                    num_collected=seq_length,
                                    num_bytes=message_bytes,
                                    seq_length=seq_length,
                                    num_features=num_features)

    print_table(standard_breakdown, group_breakdown)

