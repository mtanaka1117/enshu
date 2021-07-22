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

THRESHOLD_FACTOR = 1.1

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


def get_operation_energy(energy_readings: OrderedDict, baseline_power: float, num_seq: int) -> Tuple[List[float], List[EnergyRange]]:
    """
    Returns the start and end time of the N longest contiguous ranges of high power.

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

    power_threshold = THRESHOLD_FACTOR * baseline_power

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

            baseline_energy = baseline_power * time_diff
            op_energy -= baseline_energy
            
            # Append to the energy list
            energy_list.append(op_energy)
            ranges.append((start_time, end_time))

            # Reset the range variables
            start_time = None
            end_time = None

    # Get the top N ranges by length
    sorted_indices = np.argsort([-1 * x for x in energy_list])
    indices_to_keep = sorted_indices[:(num_seq + 1)]

    top_ranges = [EnergyRange(start=ranges[i][0], end=ranges[i][1], energy=energy_list[i]) for i in sorted(indices_to_keep)]
    return top_ranges


#def should_ignore_ad(start_time: str, end_time: str, to_ignore: List[Tuple[int, int]]):
#    start_int = int(start_time)
#    end_int = int(end_time)
#
#    for r in to_ignore:
#        if ((r[0] <= start_int) and (start_int <= r[1])) or ((r[0] <= end_int) and (end_int <= r[1])):
#            return True
#
#    return False
#
#
#def get_advertisement_ranges(energy_readings: OrderedDict, comm_ranges: List[EnergyRange], baseline_power: float) -> List[EnergyRange]:
#    start_time = min(int(r.end) for r in comm_ranges)
#    end_time = max(int(r.start) for r in comm_ranges)
#
#    ranges_to_ignore = [(int(r.start), int(r.end)) for r in comm_ranges]
#
#    prev_times: deque = deque()
#    ad_start: Optional[str] = None
#    result: List[EnergyRange] = []
#
#    power_threshold = baseline_power * THRESHOLD_FACTOR
#
#    for time, record in energy_readings.items():
#        time_num = int(time)
#
#        if (time_num > start_time) and (time_num < end_time):
#            power = record.current * record.voltage  # The power in mW
#
#            if (ad_start is None) and (power >= power_threshold):
#                ad_start = prev_times[0]
#            elif (ad_start is not None) and (power < power_threshold):
#                ad_end = str(time_num)
#                time_diff = (int(ad_end) - int(ad_start)) / 1e9
#                baseline_energy = time_diff * baseline_power
#
#                energy_diff = energy_readings[ad_end].energy - energy_readings[ad_start].energy
#                ad_energy = energy_diff - baseline_energy
#
#                if not should_ignore_ad(start_time=ad_start, end_time=ad_end, to_ignore=ranges_to_ignore):
#                    record = EnergyRange(start=ad_start,
#                                         end=ad_end,
#                                         energy=ad_energy)
#
#                    result.append(record)
#
#                ad_start = None
#
#        prev_times.append(time)
#        while len(prev_times) > 5:
#            prev_times.popleft()
#
#    return result


def breakdown_energy(energy_readings: OrderedDict, comm_ranges: List[EnergyRange]) -> Tuple[List[float], List[float]]:

    op_energy: List[float] = []
    comm_energy: List[float] = []

    for op_idx in range(1, len(comm_ranges)):
        prev_range = comm_ranges[op_idx - 1]
        curr_range = comm_ranges[op_idx]

        start_record = energy_readings[prev_range.end]
        end_record = energy_readings[curr_range.end]

        op_energy.append(end_record.energy - start_record.energy)
        comm_energy.append(curr_range.energy)

    return op_energy, comm_energy        


    #start_time = min(int(r.end) for r in comm_ranges)
    #end_time = max(int(r.end) for r in comm_ranges)
    #time_diff = (end_time - start_time) / 1e9

    #ad_energy = sum((r.energy for r in ad_ranges))

    #op_energy = energy_readings[str(end_time)].energy - energy_readings[str(start_time)].energy
    #comm_energy = sum((r.energy for i, r in enumerate(comm_ranges) if i > 0))

    #baseline_energy = time_diff * baseline_power
    #comp_energy = op_energy - comm_energy - ad_energy - baseline_energy

    #return op_energy, comm_energy, comp_energy


def get_baseline_power(energy_readings: OrderedDict) -> float:
    power = [r.current * r.voltage for r in energy_readings.values()]
    return min(filter(lambda p: p > 0, power))


def plot(energy_readings: OrderedDict, comm_ranges: List[EnergyRange], output_path: str):
    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots()

        times = [t for i, t in enumerate(energy_readings.keys()) if (i % 10) == 0]
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

        plt.savefig(output_path)
        #plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--num-seq', type=int, required=True)
    args = parser.parse_args()

    # Hold energy values
    op_energy_list: List[float] = []
    comm_energy_list: List[float] = []
    comp_energy_list: List[float] = []
    baseline_power_list: List[float] = []

    for path in iterate_dir(args.folder, '.*csv'):
        energy_readings = read_trace_file(path)

        # Get the baseline power for this trace
        baseline = get_baseline_power(energy_readings=energy_readings)

        comm_ranges = get_operation_energy(energy_readings=energy_readings,
                                           num_seq=args.num_seq,
                                           baseline_power=baseline)

        #ad_ranges = get_advertisement_ranges(energy_readings=energy_readings,
        #                                     comm_ranges=comm_ranges,
        #                                     baseline_power=baseline)

        op_energy, comm_energy = breakdown_energy(energy_readings=energy_readings,
                                                  comm_ranges=comm_ranges)

        # Log the energy values for operations and communication
        op_energy_list.extend(op_energy)
        comm_energy_list.extend(comm_energy)
        #comp_energy_list.append(comp_energy)
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
        'op_energy': op_energy_list,
        'comm_energy': comm_energy_list,
        'baseline_power': baseline_power_list,
        'num_seq': args.num_seq
    }

    save_json(result_dict, output_path)
