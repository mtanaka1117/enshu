import numpy as np
import math
from typing import List, Tuple
from sklearn.metrics import mean_absolute_error

from adaptiveleak.utils.data_utils import array_to_float, array_to_fp, num_bits_for_value
from adaptiveleak.utils.constants import BIG_NUMBER, BITS_PER_BYTE


class ShiftGroup:
    
    def __init__(self, shift: int, count: int, group_id: int, parent: int):
        self.parent = parent
        self.shift = shift
        self.count = count
        self.group_id = group_id

    def __str__(self) -> str:
        return '(Id: {0}, P: {1}, S: {2}, C: {3})'.format(self.group_id, self.parent, self.shift, self.count)

def absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.average(np.abs(y_true - y_pred))


class UnionFind:

    def __init__(self, group_shifts: List[int], reps: List[int]):
        union_find: List[ShiftGroup] = []

        start_idx = 0
        for group_id, (shift, rep) in enumerate(zip(group_shifts, reps)):
            end_idx = start_idx + rep

            group = ShiftGroup(shift=shift,
                               count=end_idx - start_idx,
                               parent=-1,
                               group_id=group_id)

            union_find.append(group)
            start_idx += rep

        self._union_find = union_find
        self._num_groups = len(union_find)

    def get_num_groups(self) -> int:
        return self._num_groups

    def get(self, idx: int) -> ShiftGroup:
        return self._union_find[idx]

    def find(self, group: ShiftGroup) -> ShiftGroup:
        group_iter = group
        while (group_iter.parent != -1):
            group_iter = self._union_find[group_iter.parent]

        return group_iter

    def union(self, g1: ShiftGroup, g2: ShiftGroup):
        p1: ShiftGroup = self.find(g1)
        p2: ShiftGroup = self.find(g2)

        if (p1.group_id == p2.group_id):
            return
        elif (p1.group_id < p2.group_id):
            left_parent = p1
            right_parent = p2
        else:
            left_parent = p2
            right_parent = p1

        right_parent.parent = left_parent.group_id
        left_parent.count += right_parent.count
        left_parent.shift = max(left_parent.shift, right_parent.shift)
        self._num_groups -= 1

    def get_groups_to_merge(self, num_to_merge: int) -> List[int]:
        scores: List[float] = []
        left_parents: List[int] = []

        for group_id in range(len(self._union_find) - 1):
            left = self._union_find[group_id]
            right = self._union_find[group_id + 1]

            shift_diff = 2 * abs(left.shift - right.shift)
            score = (left.count + right.count + shift_diff) * int(shift_diff > 0)

            if (len(scores) < num_to_merge) or (score < scores[num_to_merge - 1]):

                if len(scores) < num_to_merge:
                    scores.append(score)
                    left_parents.append(left.group_id)
                else:
                    scores[num_to_merge - 1] = score
                    left_parents[num_to_merge - 1] = left.group_id

                # Use an insertion sort step to re-sort the scores list
                i = len(scores) - 1
                while (i > 0) and (scores[i] < scores[i-1]):
                    temp = scores[i-1]
                    scores[i-1] = scores[i]
                    scores[i] = temp

                    temp = left_parents[i-1]
                    left_parents[i-1] = left_parents[i]
                    left_parents[i] = temp

                    i -= 1

            group_id += 1

        return left_parents

    def get_parents(self) -> List[ShiftGroup]:
        return list(filter(lambda g: g.parent == -1, self._union_find))
        
    def __str__(self) -> str:
        return ';'.join(map(str, self._union_find))


def merge_shift_groups(values: List[float], shifts: List[int], max_num_groups: int) -> Tuple[List[int], List[int]]:
    """
    Merges the given shift groups to meet the given budget in a manner
    which minimizes the induced error.

    Args:
        values: A list of the measurement values
        shifts: A list of the current per-element shifts
        max_num_groups: The maximum number of groups to allow (K)
    Returns:
        A pair of length-K lists denoting the shifts and repetitions
    """
    # Create the initial groups using run-length encoding
    grouped_shifts, reps = compute_runs(shifts)

    # We calculate the size required if we encoded all measurements fully
    # with the current group construction. If this meets the budget,
    # then we can stop here.
    #size_width = num_bits_for_value(max(reps))
    #size_bytes = int(math.ceil((size_width * len(reps)) / BITS_PER_BYTE))
    #total_bytes = size_bytes + len(reps) + 1

    #for group_size in reps:
    #    total_bytes += int(math.ceil((group_size * width) / BITS_PER_BYTE))

    if (len(grouped_shifts) <= max_num_groups):
        return grouped_shifts, reps

    # Initialize the union-find structure
    union_find = UnionFind(group_shifts=grouped_shifts, reps=reps)

    # Get the groups to merge
    num_to_merge = len(grouped_shifts) - max_num_groups
    groups_to_merge = union_find.get_groups_to_merge(num_to_merge=num_to_merge)

    # Merge the given groups
    for left_idx in groups_to_merge:
        left = union_find.get(left_idx)
        right = union_find.get(left_idx + 1)

        union_find.union(left, right)

    # Get all of the parents
    final_groups = union_find.get_parents()

    # Reconstruct the shifts and repetitions
    merged_shifts: List[int] = []
    merged_reps: List[int] = []

    for group in final_groups:
        merged_shifts.append(group.shift)
        merged_reps.append(group.count)

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
