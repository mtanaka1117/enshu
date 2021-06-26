import os.path
import csv
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import namedtuple, OrderedDict
from typing import List, Dict, Tuple

from adaptiveleak.utils.file_utils import save_json, iterate_dir, read_json

# Records the current (mA), voltage (V), and energy (mJ)
TraceRecord = namedtuple('TraceRecord', ['current', 'voltage', 'energy'])
THRESHOLD_PERCENTILE = 20


def read_trace_file(path: str) -> OrderedDict:
    result: OrderedDict = OrderedDict()

    with open(path, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for idx, line in enumerate(reader):
            if idx > 0:
                t = line[0]
                current = float(line[1]) / 1e6
                voltage = float(line[2]) / 1e3
                energy = float(line[-1]) / 1e3

                result[t] = TraceRecord(current=current,
                                        voltage=voltage,
                                        energy=energy)
    return result


def get_threshold(energy_readings: OrderedDict):
    # Get the unique energy values
    unique_values = set((r.current for r in energy_readings.values()))

    # Return the corresponding percentile
    return np.percentile(list(unique_values), q=THRESHOLD_PERCENTILE) * 3.3


def get_operation_energy(energy_readings: OrderedDict, threshold: float, num_trials: int) -> Tuple[List[float], List[Tuple[str, str]]]:
    """
    Returns the start and end time of the N longest contiguous ranges of higher power.

    Args:
        energy_readings: A dictionary mapping time -> trace record
        threshold: The threshold which defines a 'high' power region
        num_trials: The number of operation iterations (N)
    Returns:
        A list of at most N energy values (mJ), one per operation iteration
    """
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    energy_list: List[float] = []
    ranges: List[Tuple[str, str]] = []

    for time, record in energy_readings.items():
        power = record.current * record.voltage  # The current power (mW)

        if (start_time is None) and (power > threshold):
            start_time = time
        elif (start_time is not None) and (power <= threshold):
            end_time = time

            start_energy = energy_readings[start_time].energy
            end_energy = energy_readings[end_time].energy

            time_diff = (float(end_time) - float(start_time)) / 1e9  # Time in seconds
            baseline_energy = threshold * time_diff  # Energy required by baseline device operation

            total_energy = end_energy - start_energy
            op_energy = total_energy - baseline_energy

            # Append to the energy list
            energy_list.append(op_energy)
            ranges.append((start_time, end_time))

            # Reset the range variables
            start_time = None
            end_time = None

    if len(energy_list) <= num_trials:
        return energy_list, ranges

    # Get the top N ranges by length
    sorted_indices = np.argsort([-1 * x for x in energy_list])
    indices_to_keep = sorted_indices[:num_trials]

    top_energy = [energy_list[i] for i in sorted(indices_to_keep)]
    top_ranges = [ranges[i] for i in sorted(indices_to_keep)]

    return top_energy, top_ranges


def get_energy_per_operation(energy_readings: OrderedDict, op_range: Tuple[str, str], ops_per_trial: int, threshold: float) -> float:
    start, end = op_range

    start_energy = energy_readings[start].energy  # Starting energy (mJ)
    end_energy = energy_readings[end].energy  # Ending energy (mJ)

    time_range = (float(end) - float(start)) / 1e9  # Time in seconds
    baseline_energy = threshold * time_range  # Energy from baseline device operation

    total_energy = (end_energy - start_energy)
    total_energy -= baseline_energy

    return total_energy / float(ops_per_trial)


def plot(energy_readings: OrderedDict, threshold: float, ranges: List[Tuple[str, str]], output_path: str):
    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots()

        times = [t for i, t in enumerate(energy_readings.keys()) if (i % 100) == 0]
        power = [(energy_readings[t].current * energy_readings[t].voltage) for t in times]

        xs = list(map(lambda t: int(t) / 1e6, times))

        ax.plot(xs, power, linewidth=3)

        ax.axhline(threshold, color='tab:orange', linewidth=1)

        for op_range in ranges:
            start = int(op_range[0]) / 1e6
            end = int(op_range[1]) / 1e6

            ax.axvline(x=start, color='k', linewidth=1)
            ax.axvline(x=end, color='tab:red', linewidth=1)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Power (mW)')
        ax.set_title('Device Power over Time')

        plt.savefig(output_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    args = parser.parse_args()

    # Read the meta-data from this experiment
    metadata = read_json(os.path.join(args.folder, 'metadata.json'))
    num_trials_list: List[int] = metadata['num_trials']
    ops_per_trial: int = int(metadata['ops_per_trial'])


    energy_list: List[float] = []

    for idx, num_trials in enumerate(num_trials_list):
        path = os.path.join(args.folder, 'trial_{0}.csv'.format(idx))
        energy_readings = read_trace_file(path)
        power_threshold = get_threshold(energy_readings)

        op_energy, ranges = get_operation_energy(energy_readings=energy_readings,
                                                 threshold=power_threshold,
                                                 num_trials=num_trials)

        # Divide the energy values by the number of operations in each iteration
        energy_list.extend((e / ops_per_trial) for e in op_energy)

        # Plot the energy values
        plot(energy_readings=energy_readings,
             threshold=power_threshold,
             ranges=ranges,
             output_path=os.path.join(args.folder, 'trial_{0}.pdf'.format(idx)))

    # Save the energy readings
    output_path = os.path.join(args.folder, 'energy.json')

    avg = float(np.average(energy_list))
    std = float(np.std(energy_list))

    result_dict = {
        'avg': avg,
        'std': std,
        'energy': energy_list
    }

    save_json(result_dict, output_path)
