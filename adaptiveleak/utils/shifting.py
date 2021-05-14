import numpy as np
from typing import List, Tuple
from sklearn.metrics import mean_absolute_error

from adaptiveleak.utils.data_utils import array_to_float, array_to_fp
from adaptiveleak.utils.constants import BIG_NUMBER


class ShiftGroup:
    
    def __init__(self, shift: int, start_idx: int, end_idx: int, group_id: int, parent: int, next_parent: int):
        self.parent = parent
        self.next_parent = next_parent
        self.shift = shift
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.group_id = group_id

    @property
    def count(self) -> int:
        return self.end_idx - self.start_idx

    def __str__(self) -> str:
        return '(Id: {0}, P: {1}, N: {2}, S: {3}, C: {4})'.format(self.group_id, self.parent, self.next_parent, self.shift, self.count)

def absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.average(np.abs(y_true - y_pred))


class UnionFind:

    def __init__(self, group_shifts: List[int], reps: List[int]):
        union_find: List[ShiftGroup] = []

        start_idx = 0
        for group_id, (shift, rep) in enumerate(zip(group_shifts, reps)):
            end_idx = start_idx + rep

            group = ShiftGroup(shift=shift,
                               start_idx=start_idx,
                               end_idx=end_idx,
                               parent=-1,
                               next_parent=group_id + 1,
                               group_id=group_id)

            union_find.append(group)
            start_idx += rep

        self._union_find = union_find
        self._num_groups = len(union_find)

    def get_num_groups(self) -> int:
        return self._num_groups

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
        left_parent.end_idx = right_parent.end_idx
        left_parent.shift = max(left_parent.shift, right_parent.shift)
        left_parent.next_parent = right_parent.next_parent
        self._num_groups -= 1

    def get_groups_to_merge(self) -> Tuple[ShiftGroup, ShiftGroup]:
        # Locate the first parent
        group_id = 0
        while (self._union_find[group_id].parent != -1):
            group_id += 1

        left = self._union_find[group_id]
        right = self._union_find[left.next_parent]

        if (left.shift == right.shift):
            return left, right

        best_id = group_id
        best_score = (left.count + right.count) + abs(left.shift - right.shift)

        group_id = left.next_parent
        while (group_id < len(self._union_find) and self._union_find[group_id].next_parent < len(self._union_find)):
            left = self._union_find[group_id]
            right = self._union_find[left.next_parent]

            if (left.shift == right.shift):
                return left, right

            score = (left.count + right.count) + abs(left.shift - right.shift)

            if (score < best_score):
                best_id = group_id
                best_score = score

            group_id = left.next_parent

        best_left = self._union_find[best_id]
        best_right = self._union_find[best_left.next_parent]
        return best_left, best_right

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

    if len(grouped_shifts) <= max_num_groups:
        return grouped_shifts, reps

    # Initialize the union-find structure
    union_find = UnionFind(group_shifts=grouped_shifts, reps=reps)

    while (union_find.get_num_groups() > max_num_groups):
        left, right = union_find.get_groups_to_merge()
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



