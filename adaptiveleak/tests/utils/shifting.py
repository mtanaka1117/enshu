import unittest
import numpy as np

from adaptiveleak.utils.shifting import merge_shift_groups


class TestShiftMerging(unittest.TestCase):

    def test_one_merge_first(self):
        values = np.array([0.5, 0.75, 0.125, 0.375])
        shifts = [0, -1, -2, -2]

        merged, reps = merge_shift_groups(values=values,
                                          shifts=shifts,
                                          max_num_groups=2)

        self.assertEqual(merged, [0, -2])
        self.assertEqual(reps, [2, 2])

    def test_one_merge_second(self):
        values = np.array([2.5, 2.5, 0.26, 0.375])
        shifts = [0, 0, -1, -2]

        merged, reps = merge_shift_groups(values=values,
                                          shifts=shifts,
                                          max_num_groups=2)

        self.assertEqual(merged, [0, -1])
        self.assertEqual(reps, [2, 2])

    def test_one_merge_left(self):
        values = np.array([2.5, 2.5, 0.125, 0.75, 0.26])
        shifts = [0, 0, -2, -1, -1]

        merged, reps = merge_shift_groups(values=values,
                                          shifts=shifts,
                                          max_num_groups=2)

        self.assertEqual(merged, [0, -1])
        self.assertEqual(reps, [2, 3])

    def test_two_merges(self):
        values = np.array([1.5, 1.5, 0.125, 0.75, 0.25, 0.125, 0.375])
        shifts = [0, 0, -2, -3, -3, -2, -2]

        merged, reps = merge_shift_groups(values=values,
                                          shifts=shifts,
                                          max_num_groups=2)

        self.assertEqual(merged, [0, -2])
        self.assertEqual(reps, [2, 5])

    def test_two_merge_cascade(self):
        values = np.array([2.5, 2.5, 1.125, 0.0625, 1.125, 1.375])
        shifts = [0, 0, -2, -3, -2, -2]

        merged, reps = merge_shift_groups(values=values,
                                          shifts=shifts,
                                          max_num_groups=2)

        self.assertEqual(merged, [0, -2])
        self.assertEqual(reps, [2, 4])


if __name__ == '__main__':
    unittest.main()
