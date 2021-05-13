import numpy as np
from typing import List, Tuple
from sklearn.metrics import mean_absolute_error

from adaptiveleak.utils.data_utils import array_to_float, array_to_fp
from adaptiveleak.utils.constants import BIG_NUMBER


class GroupEntry:
    
    def __init__(self, error: float, left_error: float, right_error: float, shift: int, start_idx: int, end_idx: int):
        self.error = error
        self.left_error = left_error
        self.right_error = right_error
        self.shift = shift
        self.start_idx = start_idx
        self.end_idx = end_idx


def absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.average(np.abs(y_true - y_pred))


def merge_shift_groups(values: List[float], shifts: List[int], width: int, precision: int, max_num_groups: int) -> Tuple[List[int], List[int]]:
    """
    Merges the given shift groups to meet the given budget in a manner
    which minimizes the induced error.

    Args:
        values: A list of the measurement values
        shifts: A list of the current per-element shifts
        width: The bit-width of each measurement
        precision: The base precision of each measurement
        max_num_groups: The maximum number of groups to allow (K)
    Returns:
        A pair of length-K lists denoting the shifts and repetitions
    """
    # Create the initial groups using run-length encoding
    grouped_shifts, reps = compute_runs(shifts)

    print('Starting Runs: {0}'.format(len(reps)))

    if len(grouped_shifts) <= max_num_groups:
        return grouped_shifts, reps

    # Create the initial groups
    groups: List[GroupEntry] = []
    start_idx = 0
    for shift, rep in zip(grouped_shifts, reps):
        group_values = values[start_idx:start_idx+rep]
        fixed_point = array_to_fp(group_values, width=width, precision=precision - shift)
        quantized = array_to_float(fixed_point, precision=precision - shift)

        error = absolute_error(y_true=group_values, y_pred=quantized)

        entry = GroupEntry(error=error,
                           right_error=BIG_NUMBER,
                           left_error=BIG_NUMBER,
                           shift=shift,
                           start_idx=start_idx,
                           end_idx=start_idx + rep)
        groups.append(entry)

        start_idx += rep

    # Initialize the error terms
    for i in range(len(groups)):
        group = groups[i]
        group_values = values[group.start_idx:group.end_idx]

        if i < len(groups) - 1:
            right_shift = groups[i+1].shift
            fixed_point = array_to_fp(group_values, width=width, precision=precision - right_shift)
            quantized = array_to_float(fixed_point, precision=precision - right_shift)

            group.right_error = absolute_error(y_true=group_values, y_pred=quantized)

        if i > 0:
            left_shift = groups[i-1].shift
            fixed_point = array_to_fp(group_values, width=width, precision=precision - left_shift)
            quantized = array_to_float(fixed_point, precision=precision - left_shift)

            group.left_error = absolute_error(y_true=group_values, y_pred=quantized)

    # Combine elements with minimum error until we reach
    # the target size
    while len(groups) > max_num_groups:
        right_errors = [(g.right_error - g.error) for g in groups]
        left_errors = [(g.left_error - g.error) for g in groups]

        right_min_idx = np.argmin(right_errors)
        left_min_idx = np.argmin(left_errors)

        right_min_error = right_errors[right_min_idx]
        left_min_error = left_errors[left_min_idx]

        # Merge the groups
        if right_min_error < left_min_error:
            min_group = groups[right_min_idx]
            right_group = groups[right_min_idx + 1]
            min_group_elem = min_group.end_idx - min_group.start_idx

            min_group.end_idx = right_group.end_idx
            min_group.shift = right_group.shift
            merged_values = values[min_group.start_idx:min_group.end_idx]

            if right_min_idx < len(groups) - 2:
                right_shift = groups[right_min_idx + 2].shift
                fixed_point = array_to_fp(merged_values, width=width, precision=precision - right_shift)
                quantized = array_to_float(fixed_point, precision=precision - right_shift)

                min_group.right_error = absolute_error(y_true=merged_values, y_pred=quantized)
            else:
                min_group.right_error = BIG_NUMBER

            # Update the previous entry's right error and current entry's left error if needed
            if right_min_idx > 0:
                prev_group = groups[right_min_idx - 1]
                
                prev_values = values[prev_group.start_idx:prev_group.end_idx]
                fixed_point = array_to_fp(prev_values, width=width, precision=precision - min_group.shift)
                quantized = array_to_float(fixed_point, precision=precision - min_group.shift)
                prev_group.right_error = absolute_error(y_true=prev_values, y_pred=quantized)

                fixed_point = array_to_fp(merged_values, width=width, precision=precision - prev_group.shift)
                quantized = array_to_float(fixed_point, precision=precision - prev_group.shift)
                min_group.left_error = absolute_error(y_true=merged_values, y_pred=quantized)
            else:
                min_group.left_error = BIG_NUMBER

            right_group_elem = right_group.end_idx - right_group.start_idx
            min_group.error = (min_group_elem * min_group.right_error + right_group_elem * right_group.error) / (min_group_elem + right_group_elem)

            groups.pop(right_min_idx + 1)
        else:
            min_group = groups[left_min_idx]
            left_group = groups[left_min_idx - 1]
            min_group_elem = min_group.end_idx - min_group.start_idx

            min_group.start_idx = left_group.start_idx
            min_group.shift = left_group.shift
            merged_values = values[min_group.start_idx:min_group.end_idx]

            if left_min_idx > 1:
                left_shift = groups[left_min_idx - 2].shift
                fixed_point = array_to_fp(merged_values, width=width, precision=precision - left_shift)
                quantized = array_to_float(fixed_point, precision=precision - left_shift)

                min_group.left_error = absolute_error(y_true=merged_values, y_pred=quantized)
            else:
                min_group.left_error = BIG_NUMBER

            # Update the next group's left error and current group's right error if needed
            if left_min_idx < len(groups) - 1:
                next_group = groups[left_min_idx + 1]
                
                next_values = values[next_group.start_idx:next_group.end_idx]
                fixed_point = array_to_fp(next_values, width=width, precision=precision - min_group.shift)
                quantized = array_to_float(fixed_point, precision=precision - min_group.shift)

                next_group.left_error = absolute_error(y_true=next_values, y_pred=quantized)

                fixed_point = array_to_fp(merged_values, width=width, precision=precision - next_group.shift)
                quantized = array_to_float(fixed_point, precision=precision - next_group.shift)
                min_group.right_error = absolute_error(y_true=merged_values, y_pred=quantized)
            else:
                min_group.right_error = BIG_NUMBER

            left_group_elem = (left_group.end_idx - left_group.start_idx)
            min_group.error = (min_group_elem * min_group.left_error + left_group_elem * left_group.error) / (min_group_elem + left_group_elem)

            groups.pop(left_min_idx - 1)

        #merged_values = values[min_group.start_idx:min_group.end_idx]
        #fixed_point = array_to_fp(merged_values, width=width, precision=precision - min_group.shift)
        #quantized = array_to_float(fixed_point, precision=precision - min_group.shift)

        #min_group.error = absolute_error(y_true=merged_values, y_pred=quantized)

    # Convert the groups back into shifts and reps
    merged_shifts = [g.shift for g in groups]
    merged_reps = [(g.end_idx - g.start_idx) for g in groups]

    return merged_shifts, merged_reps


def compute_runs(values: List[int]) -> Tuple[List[int], List[int]]:
    current = values[0]
    current_count = 1

    encoded: List[int] = []
    reps: List[int] = []

    for i in range(1, len(values)):
        val = values[i]

        if (val != current):
            encoded.append(current)
            reps.append(current_count)

            current = val
            current_count = 1
        else:
            current_count += 1

    # Always include the final element
    encoded.append(current)
    reps.append(current_count)

    return encoded, reps



