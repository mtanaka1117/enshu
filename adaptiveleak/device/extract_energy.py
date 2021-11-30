import os.path
import csv
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import namedtuple, OrderedDict, deque
from typing import List, Dict, Tuple

from adaptiveleak.utils.constants import BIG_NUMBER
from adaptiveleak.utils.file_utils import save_json, iterate_dir, read_json


# Records the current (mA), voltage (V), and energy (mJ)
TraceRecord = namedtuple('TraceRecord', ['current', 'voltage', 'energy'])
EnergyRange = namedtuple('EnergyRange', ['start', 'end', 'energy'])

POWER_THRESHOLD = 2  # in mW
BASELINE_POWER = 0.6  # Set a fixed baseline power to smooth over any relative errors we see


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


def get_communication_energy(energy_readings: OrderedDict, num_seq: int) -> Tuple[List[float], List[EnergyRange]]:
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

    energy_list: List[float] = []
    ranges: List[Tuple[str, str]] = []

    for time, record in energy_readings.items():
        power = record.current * record.voltage  # The current power (mW)

        if (start_time is None) and (power >= POWER_THRESHOLD):
            start_time = time
        elif (start_time is not None) and (power <= POWER_THRESHOLD):
            end_time = time

            start_energy = energy_readings[start_time].energy
            end_energy = energy_readings[end_time].energy
            op_energy = end_energy - start_energy

            # Append to the energy list
            energy_list.append(op_energy)
            ranges.append((start_time, end_time))

            # Reset the range variables
            start_time = None
            end_time = None

    # Get the top (N + 1) ranges by length
    sorted_indices = np.argsort([-1 * x for x in energy_list])
    indices_to_keep = sorted_indices[:(num_seq + 1)]

    top_ranges = [EnergyRange(start=ranges[i][0], end=ranges[i][1], energy=energy_list[i]) for i in sorted(indices_to_keep)]
    return top_ranges


def get_energy_per_seq(energy_readings: OrderedDict, comm_ranges: List[EnergyRange]) -> List[float]:
    """
    Gets the energy for each operation.

    Args:
        energy_readings: A dictionary mapping time -> trace record
        comm_ranges: An (ordered) list of the communication events
    Returns:
        A list of [N] energy readings where N is the number of sequences
    """
    result: List[float] = []

    for seq_idx in range(len(comm_ranges) - 1):
        prev_range = comm_ranges[seq_idx]
        curr_range = comm_ranges[seq_idx + 1]

        start_time = prev_range.end  # Start at the end of the previous communication event
        end_time = curr_range.end  # End at the end of the current communication event

        energy = energy_readings[end_time].energy - energy_readings[start_time].energy
        result.append(energy)

    return result


def get_baseline_power(energy_readings: OrderedDict) -> float:
    # Get the minimum power for all current readings above 0. Often, EnergyTrace
    # will report 0 current at the beginning of the trace.
    min_power = BIG_NUMBER

    for t, record in energy_readings.items():
        if (record.current > 0):
            min_power = min(min_power, record.current * record.voltage)

    return min_power


def plot(energy_readings: OrderedDict, comm_ranges: List[EnergyRange], output_path: str):
    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots()

        times = [t for i, t in enumerate(energy_readings.keys()) if (i % 100) == 0]
        power = [(energy_readings[t].current * energy_readings[t].voltage) for t in times]

        xs = list(map(lambda t: int(t) / 1e6, times))

        ax.plot(xs, power, linewidth=3)

        baseline = get_baseline_power(energy_readings)
        ax.axhline(baseline, color='tab:orange', linewidth=1)

        for op_range in comm_ranges:
            start = int(op_range.start) / 1e6
            end = int(op_range.end) / 1e6

            ax.axvline(x=start, color='k', linewidth=1)
            ax.axvline(x=end, color='tab:red', linewidth=1)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Power (mW)')
        ax.set_title('Device Power over Time')

        plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='The folder containing the EnergyTrace CSV files.')
    parser.add_argument('--num-seq', type=int, required=True, help='The number of sequences in the experiment.')
    args = parser.parse_args()

    # Hold energy values
    energy_list: List[float] = []
    seq_energy_list: List[float] = []
    time_list: List[int] = []
    baseline_power_list: List[float] = []

    for path in iterate_dir(args.folder, '.*csv'):
        energy_readings = read_trace_file(path)

        # Get the baseline power for this trace
        baseline = get_baseline_power(energy_readings=energy_readings)

        comm_ranges = get_communication_energy(energy_readings=energy_readings,
                                               num_seq=args.num_seq)

        # Get the energy per sequence
        energy_per_seq = get_energy_per_seq(energy_readings=energy_readings,
                                            comm_ranges=comm_ranges)

        total_energy = np.sum(energy_per_seq)

        # Get the total time for the experiment
        start_time = min(int(r.end) for r in comm_ranges)
        end_time = max(int(r.end) for r in comm_ranges)
        time_window = (end_time - start_time) / 1e9

        # Log the energy values for operations and communication
        energy_list.append(total_energy)
        seq_energy_list.extend(energy_per_seq)
        time_list.append(time_window)
        baseline_power_list.append(baseline)

        # Plot the energy values
        file_name = os.path.basename(path)
        file_name = file_name.split('.')[0]

        plot(energy_readings=energy_readings,
             comm_ranges=comm_ranges,
             output_path=os.path.join(args.folder, '{0}.pdf'.format(file_name)))

    # Save the energy readings
    output_path = os.path.join(args.folder, 'energy.json')

    result_dict = {
        'total_energy': energy_list,
        'avg_energy_per_seq': float(np.average(seq_energy_list)),
        'std_energy_per_seq': float(np.std(seq_energy_list)),
        'median_energy_per_seq': float(np.median(seq_energy_list)),
        'seq_energy': seq_energy_list,
        'baseline_power': baseline_power_list,
        'time': time_list,
        'num_seq': args.num_seq
    }

    save_json(result_dict, output_path)
