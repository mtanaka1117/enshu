import os.path
import csv
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import namedtuple, OrderedDict, deque
from typing import List, Dict, Tuple

from adaptiveleak.utils.file_utils import save_json, iterate_dir, read_json

# Records the current (mA), voltage (V), and energy (mJ)
TraceRecord = namedtuple('TraceRecord', ['current', 'voltage', 'energy'])
EnergyRange = namedtuple('EnergyRange', ['start', 'end', 'energy'])

POWER_THRESHOLD = 0.3  # in mW


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


def get_operation_energy(energy_readings: OrderedDict, active_power: float, num_trials: int) -> Tuple[List[float], List[EnergyRange]]:
    """
    Returns the start and end time of the N longest contiguous ranges of higher power.

    Args:
        energy_readings: A dictionary mapping time -> trace record
        num_seq: The number of processed sequences (N)
    Returns:
        A list of N + 1 energy values (mJ), one per operation iteration
    """
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    #power_threshold = 1.05 * active_power
    power_threshold = POWER_THRESHOLD

    energy_list: List[float] = []
    ranges: List[Tuple[str, str]] = []

    for time, record in energy_readings.items():
        power = record.current * record.voltage  # The current power (mW)

        if (start_time is None) and (power >= power_threshold):
            start_time = time
        elif (start_time is not None) and (power <= power_threshold):
            end_time = time

            start_energy = energy_readings[start_time].energy
            end_energy = energy_readings[end_time].energy
            op_energy = end_energy - start_energy

            time_diff = (float(end_time) - float(start_time)) / 1e9  # Time in seconds

            active_energy = active_power * time_diff
            op_energy -= active_energy
            
            # Append to the energy list
            energy_list.append(op_energy)
            ranges.append((start_time, end_time))

            # Reset the range variables
            start_time = None
            end_time = None

    # Get the top N ranges by length
    sorted_indices = np.argsort([-1 * x for x in energy_list])
    indices_to_keep = sorted_indices[:num_trials]

    top_ranges = [EnergyRange(start=ranges[i][0], end=ranges[i][1], energy=energy_list[i]) for i in sorted(indices_to_keep)]
    return top_ranges


def get_comm_energy(energy_readings: OrderedDict, comm_ranges: List[EnergyRange]) -> Tuple[List[float], float]:
    comm_energy_list: List[float] = []
    #comm_time_list: List[float] = []

    for comm_range in comm_ranges:
        start_time = int(comm_range.start)
        end_time = int(comm_range.end)

        comm_energy = comm_range.energy
        comm_energy_list.append(comm_energy)

    max_time = max(map(int, energy_readings.keys()))
    total_energy = energy_readings[str(max_time)].energy

    total_time = max_time / 1e9
    baseline_power = (total_energy - sum(comm_energy_list)) / total_time

    return comm_energy_list, baseline_power


def get_active_power(energy_readings: OrderedDict) -> float:
    power = [r.current * r.voltage for r in energy_readings.values()]
    return min(filter(lambda p: p > 0, power))


def plot(energy_readings: OrderedDict, comm_ranges: List[EnergyRange], output_path: str):
    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots()

        times = [t for i, t in enumerate(energy_readings.keys()) if (i % 10) == 0]
        power = [(energy_readings[t].current * energy_readings[t].voltage) for t in times]

        xs = list(map(lambda t: int(t) / 1e6, times))

        ax.plot(xs, power, linewidth=3)

        active_power = get_active_power(energy_readings)
        ax.axhline(active_power, color='tab:orange', linewidth=1)

        for op_range in comm_ranges:
            start = int(op_range.start) / 1e6
            end = int(op_range.end) / 1e6

            ax.axvline(x=start, color='k', linewidth=1)
            ax.axvline(x=end, color='tab:red', linewidth=1)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Power (mW)')
        ax.set_title('Device Power over Time')

        plt.savefig(output_path)
        #plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    args = parser.parse_args()

    # Hold energy values
    comm_energy_list: List[float] = []
    baseline_power_list: List[float] = []
    active_power_list: List[float] = []

    for path in iterate_dir(args.folder, '.*csv'):
        energy_readings = read_trace_file(path)

        # Get the device active power for this trace
        active_power = get_active_power(energy_readings=energy_readings)

        comm_ranges = get_operation_energy(energy_readings=energy_readings,
                                           num_trials=5,
                                           active_power=active_power)

        comm_energy, baseline_power = get_comm_energy(energy_readings=energy_readings,
                                                      comm_ranges=comm_ranges)

        # Log the energy values for operations and communication
        comm_energy_list.extend(comm_energy)
        baseline_power_list.append(baseline_power)
        active_power_list.append(active_power)

        # Plot the energy values
        file_name = os.path.basename(path)
        file_name = file_name.split('.')[0]

        plot(energy_readings=energy_readings,
             comm_ranges=comm_ranges,
             output_path=os.path.join(args.folder, '{0}.pdf'.format(file_name)))

    # Save the energy readings
    output_path = os.path.join(args.folder, 'energy.json')

    result_dict = {
        'median': np.median(comm_energy_list),
        'comm_energy': comm_energy_list,
        'baseline_power': baseline_power_list,
        'active_power': active_power_list
    }

    save_json(result_dict, output_path)
