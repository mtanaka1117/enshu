import unittest
import numpy as np
import h5py
from sklearn.metrics import mean_absolute_error

from adaptiveleak.utils import message
from adaptiveleak.utils.constants import SMALL_NUMBER
from adaptiveleak.utils.data_utils import pad_to_length, create_groups, select_range_shifts_array
from adaptiveleak.utils.shifting import merge_shift_groups


class TestByte(unittest.TestCase):

    def test_encode_decode_six(self):
        measurements = np.array([[0.25, -0.125, 0.75], [-0.125, 0.625, -0.5]])
        precision = 6
        width = 8
        seq_length = 8
        collected_indices = [0, 3]

        encoded = message.encode_standard_measurements(measurements=measurements,
                                                       precision=precision,
                                                       width=width,
                                                       collected_indices=collected_indices,
                                                       seq_length=seq_length,
                                                       should_compress=False)

        recovered, indices, _ = message.decode_standard_measurements(byte_str=encoded,
                                                                     num_features=measurements.shape[1],
                                                                     seq_length=seq_length,
                                                                     width=width,
                                                                     precision=precision,
                                                                     should_compress=False)

        # Check recovered values
        self.assertTrue(np.all(np.isclose(measurements, recovered)))

        # Check indices
        self.assertEqual(len(indices), 2)
        self.assertEqual(indices[0], collected_indices[0])
        self.assertEqual(indices[1], collected_indices[1])

    def test_encode_decode_two(self):
        measurements = np.array([[0.25, -0.125, 0.75], [-0.125, 0.625, -0.5]])
        precision = 2
        width = 4
        seq_length = 8
        collected_indices = [0, 4]

        encoded = message.encode_standard_measurements(measurements=measurements,
                                                       precision=precision,
                                                       width=width,
                                                       collected_indices=collected_indices,
                                                       seq_length=seq_length,
                                                       should_compress=False)

        recovered, indices, _ = message.decode_standard_measurements(byte_str=encoded,
                                                                     num_features=measurements.shape[1],
                                                                     seq_length=seq_length,
                                                                     width=width,
                                                                     precision=precision,
                                                                     should_compress=False)

        expected = np.array([[0.25, 0.0, 0.75], [0.0, 0.5, -0.5]])

        # Check recovered values
        self.assertTrue(np.all(np.isclose(expected, recovered)))

        # Check indices
        self.assertEqual(len(indices), 2)
        self.assertEqual(indices[0], collected_indices[0])
        self.assertEqual(indices[1], collected_indices[1])

    def test_encode_decode_ten(self):
        measurements = np.array([[0.25, -0.125, (1.0 / 512.0)], [-0.125, 0.625, -0.5]])
        precision = 10
        width = 13
        seq_length = 8
        collected_indices = [0, 6]

        encoded = message.encode_standard_measurements(measurements=measurements,
                                                       precision=precision,
                                                       width=width,
                                                       collected_indices=collected_indices,
                                                       seq_length=seq_length,
                                                       should_compress=False)

        recovered, indices, _ = message.decode_standard_measurements(byte_str=encoded,
                                                                     num_features=measurements.shape[1],
                                                                     seq_length=seq_length,
                                                                     width=width,
                                                                     precision=precision,
                                                                     should_compress=False)

        # Check recovered values
        self.assertTrue(np.all(np.isclose(measurements, recovered)))

        # Check indices
        self.assertEqual(len(indices), 2)
        self.assertEqual(indices[0], collected_indices[0])
        self.assertEqual(indices[1], collected_indices[1])

    def test_encode_decode_two_compressed(self):
        measurements = np.array([[0.25, -0.125, 0.75], [-0.125, 0.625, -0.5]])
        precision = 2
        width = 4
        seq_length = 8
        collected_indices = [0, 4]

        encoded = message.encode_standard_measurements(measurements=measurements,
                                                       precision=precision,
                                                       width=width,
                                                       collected_indices=collected_indices,
                                                       seq_length=seq_length,
                                                       should_compress=True)

        recovered, indices, _ = message.decode_standard_measurements(byte_str=encoded,
                                                                     num_features=measurements.shape[1],
                                                                     seq_length=seq_length,
                                                                     width=width,
                                                                     precision=precision,
                                                                     should_compress=True)

        expected = np.array([[0.25, 0.0, 0.75], [0.0, 0.5, -0.5]])

        # Check recovered values
        self.assertTrue(np.all(np.isclose(expected, recovered)))

        # Check indices
        self.assertEqual(len(indices), 2)
        self.assertEqual(indices[0], collected_indices[0])
        self.assertEqual(indices[1], collected_indices[1])

    def test_encode_decode_six_compressed(self):
        measurements = np.array([[1.25, -0.125, -0.75], [1.125, -0.625, -0.5]])
        precision = 4
        width = 6
        seq_length = 8
        collected_indices = [0, 4]

        encoded = message.encode_standard_measurements(measurements=measurements,
                                                       precision=precision,
                                                       width=width,
                                                       collected_indices=collected_indices,
                                                       seq_length=seq_length,
                                                       should_compress=True)

        recovered, indices, _ = message.decode_standard_measurements(byte_str=encoded,
                                                                     num_features=measurements.shape[1],
                                                                     seq_length=seq_length,
                                                                     width=width,
                                                                     precision=precision,
                                                                     should_compress=True)

        # Check recovered values
        self.assertTrue(np.all(np.isclose(measurements, recovered)))

        # Check indices
        self.assertEqual(len(indices), 2)
        self.assertEqual(indices[0], collected_indices[0])
        self.assertEqual(indices[1], collected_indices[1])


class TestGroupWidths(unittest.TestCase):

    def test_encode_decode_widths(self):
        widths = [31, 1, 9, 12]
        shifts = [3, -4, 0, 2] 

        encoded = message.encode_group_widths(widths=widths, shifts=shifts)
        rec_widths, rec_shifts = message.decode_group_widths(encoded=encoded)

        self.assertEqual(rec_widths, widths)
        self.assertEqual(rec_shifts, shifts)

    def test_encode_decode_overflow(self):
        widths = [32, 4, 6]
        shifts = [1, 4, -5]

        encoded = message.encode_group_widths(widths=widths, shifts=shifts)
        rec_widths, rec_shifts = message.decode_group_widths(encoded=encoded)

        self.assertEqual(rec_widths, [31, 4, 6])
        self.assertEqual(rec_shifts, [1, 3, -4])


class TestStable(unittest.TestCase):

    def test_encode_decode_two_groups(self):
        measurements = np.array([[0.25, -0.125, 0.75], [-0.125, 0.625, -0.5]])
        non_fractional = 2
        seq_length = 8
        collected_indices = [0, 1]
        widths = [5, 5]
        shifts = [-2, -1]
        sizes = [3, 3]

        encoded = message.encode_stable_measurements(measurements=measurements,
                                                     collected_indices=collected_indices,
                                                     seq_length=seq_length,
                                                     widths=widths,
                                                     shifts=shifts,
                                                     group_sizes=sizes,
                                                     non_fractional=non_fractional)

        decoded, indices, widths = message.decode_stable_measurements(encoded=encoded,
                                                                      seq_length=seq_length,
                                                                      num_features=measurements.shape[1],
                                                                      non_fractional=non_fractional)

        # Check recovered values
        error = mean_absolute_error(y_true=measurements, y_pred=decoded)
        self.assertLess(error, SMALL_NUMBER)

        # Check the returned width
        self.assertEqual(widths, [5, 5])

        # Check indices
        self.assertEqual(indices, collected_indices)

    def test_encode_decode_two_groups_truncated(self):
        measurements = np.array([[0.25, -0.125, 0.75], [-0.125, 0.625, -0.5]])
        non_fractional = 2
        seq_length = 8
        collected_indices = [0, 5]
        widths = [4, 4]
        shifts = [-1, -1]
        sizes = [3, 3]

        encoded = message.encode_stable_measurements(measurements=measurements,
                                                     collected_indices=collected_indices,
                                                     seq_length=seq_length,
                                                     widths=widths,
                                                     shifts=shifts,
                                                     group_sizes=sizes,
                                                     non_fractional=non_fractional)

        decoded, indices, widths = message.decode_stable_measurements(encoded=encoded,
                                                                      seq_length=seq_length,
                                                                      num_features=measurements.shape[1],
                                                                      non_fractional=non_fractional)

        # Check recovered values
        error = mean_absolute_error(y_true=measurements, y_pred=decoded)
        self.assertLess(error, 0.03)

        # Check the widths
        self.assertEqual(widths, [4, 4])

        # Check indices
        self.assertEqual(indices, collected_indices)

    def test_encode_decode_two_groups_truncated_signed(self):
        measurements = np.array([[0.25, -0.125, -0.75], [0.125, -0.625, -0.5]])
        non_fractional = 2
        seq_length = 8
        collected_indices = [0, 5]
        widths = [4, 4]
        shifts = [-1, -1]
        sizes = [2, 4]

        encoded = message.encode_stable_measurements(measurements=measurements,
                                                     collected_indices=collected_indices,
                                                     seq_length=seq_length,
                                                     widths=widths,
                                                     shifts=shifts,
                                                     group_sizes=sizes,
                                                     non_fractional=non_fractional)

        decoded, indices, widths = message.decode_stable_measurements(encoded=encoded,
                                                                      seq_length=seq_length,
                                                                      num_features=measurements.shape[1],
                                                                      non_fractional=non_fractional)

        # Check recovered values
        error = mean_absolute_error(y_true=measurements, y_pred=decoded)
        self.assertLess(error, 0.002)

        # Check the width
        self.assertEqual(widths, [4, 4])

        # Check indices
        self.assertEqual(indices, collected_indices)

    def test_encode_decode_three_groups(self):
        measurements = np.array([[0.25, -0.125, 0.75], [-0.25, 0.625, -0.5]])
        non_fractional = 4
        seq_length = 8
        collected_indices = [0, 7]
        widths = [5, 6, 5]
        shifts = [-1, -2, 0]
        sizes = [2, 3, 1]

        encoded = message.encode_stable_measurements(measurements=measurements,
                                                     collected_indices=collected_indices,
                                                     seq_length=seq_length,
                                                     widths=widths,
                                                     shifts=shifts,
                                                     group_sizes=sizes,
                                                     non_fractional=non_fractional)

        decoded, indices, widths = message.decode_stable_measurements(encoded=encoded,
                                                                      seq_length=seq_length,
                                                                      num_features=measurements.shape[1],
                                                                      non_fractional=non_fractional)

        # Check recovered values
        error = mean_absolute_error(y_true=measurements, y_pred=decoded)
        self.assertLess(error, SMALL_NUMBER)

        # Check widths
        self.assertEqual(widths, [5, 6, 5])

        # Check indices
        self.assertEqual(indices, collected_indices)

    def test_encode_decode_small_padded(self):
        measurements = np.array([[0.25, -0.125, 0.75], [-0.125, 0.625, -0.5]])
        non_fractional = 2
        seq_length = 8
        collected_indices = [0, 1]
        widths = [5, 5]
        shifts = [-2, -1]
        sizes = [3, 3]

        encoded = message.encode_stable_measurements(measurements=measurements,
                                                     collected_indices=collected_indices,
                                                     seq_length=seq_length,
                                                     widths=widths,
                                                     shifts=shifts,
                                                     group_sizes=sizes,
                                                     non_fractional=non_fractional)

        encoded = pad_to_length(encoded, length=len(encoded) + 7)

        decoded, indices, widths = message.decode_stable_measurements(encoded=encoded,
                                                                      seq_length=seq_length,
                                                                      num_features=measurements.shape[1],
                                                                      non_fractional=non_fractional)

        # Check recovered values
        error = mean_absolute_error(y_true=measurements, y_pred=decoded)
        self.assertLess(error, SMALL_NUMBER)

        # Check the returned width
        self.assertEqual(widths, [5, 5])

        # Check indices
        self.assertEqual(indices, collected_indices)

    def test_encode_decode_large(self):
        # Load the data
        with h5py.File('../../datasets/uci_har/train/data.h5', 'r') as fin:
            inputs = fin['inputs'][0]  # [50, 6]

        width = 8
        seq_length = inputs.shape[0]
        collected_indices = list(range(seq_length))
        non_fractional = 3

        flattened = inputs.T.reshape(-1)

        # Set the shifts
        shifts = select_range_shifts_array(measurements=flattened,
                                           old_width=16,
                                           old_precision=13,
                                           new_width=width,
                                           num_range_bits=3)

        merged_shifts, sizes = merge_shift_groups(values=flattened,
                                                  shifts=shifts,
                                                  max_num_groups=6)

        # Set the widths using the number of groups
        group_widths = [width for _ in sizes]

        # Encode and Decode the message
        encoded = message.encode_stable_measurements(measurements=inputs,
                                                     collected_indices=collected_indices,
                                                     seq_length=seq_length,
                                                     widths=group_widths,
                                                     group_sizes=sizes,
                                                     shifts=merged_shifts,
                                                     non_fractional=non_fractional)

        decoded, indices, widths = message.decode_stable_measurements(encoded=encoded,
                                                                      seq_length=seq_length,
                                                                      num_features=inputs.shape[1],
                                                                      non_fractional=non_fractional)

        error = mean_absolute_error(y_true=inputs, y_pred=decoded)
        self.assertLessEqual(error, 0.01)

        self.assertEqual(widths, group_widths)

    def test_encode_decode_large_two(self):
        # Load the data
        with h5py.File('../../datasets/uci_har/train/data.h5', 'r') as fin:
            inputs = fin['inputs'][495]  # [50, 6]

        width = 8
        seq_length = inputs.shape[0]
        collected_indices = list(range(seq_length))
        non_fractional = 3

        flattened = inputs.T.reshape(-1)

        # Set the shifts
        shifts = select_range_shifts_array(measurements=flattened,
                                           old_width=16,
                                           old_precision=13,
                                           new_width=width,
                                           num_range_bits=3)

        merged_shifts, sizes = merge_shift_groups(values=flattened,
                                                  shifts=shifts,
                                                  max_num_groups=6)

        # Set the widths using the number of groups
        group_widths = [width for _ in sizes]

        # Encode and Decode the message
        encoded = message.encode_stable_measurements(measurements=inputs,
                                                     collected_indices=collected_indices,
                                                     seq_length=seq_length,
                                                     widths=group_widths,
                                                     group_sizes=sizes,
                                                     shifts=merged_shifts,
                                                     non_fractional=non_fractional)

        decoded, indices, widths = message.decode_stable_measurements(encoded=encoded,
                                                                      seq_length=seq_length,
                                                                      num_features=inputs.shape[1],
                                                                      non_fractional=non_fractional)

        error = mean_absolute_error(y_true=inputs, y_pred=decoded)
        self.assertLessEqual(error, 0.01)

        self.assertEqual(widths, group_widths)

    def test_encode_decode_large_tight(self):
        # Load the data
        with h5py.File('../../datasets/uci_har/train/data.h5', 'r') as fin:
            inputs = fin['inputs'][495]  # [50, 6]

        width = 4
        seq_length = inputs.shape[0]
        collected_indices = list(range(seq_length))
        non_fractional = 3

        flattened = inputs.T.reshape(-1)

        # Set the shifts
        shifts = select_range_shifts_array(measurements=flattened,
                                           old_width=16,
                                           old_precision=13,
                                           new_width=width,
                                           num_range_bits=3)

        merged_shifts, sizes = merge_shift_groups(values=flattened,
                                                  shifts=shifts,
                                                  max_num_groups=6)

        # Set the widths using the number of groups
        group_widths = [width for _ in sizes]

        # Encode and Decode the message
        encoded = message.encode_stable_measurements(measurements=inputs,
                                                     collected_indices=collected_indices,
                                                     seq_length=seq_length,
                                                     widths=group_widths,
                                                     group_sizes=sizes,
                                                     shifts=merged_shifts,
                                                     non_fractional=non_fractional)

        decoded, indices, widths = message.decode_stable_measurements(encoded=encoded,
                                                                      seq_length=seq_length,
                                                                      num_features=inputs.shape[1],
                                                                      non_fractional=non_fractional)

        error = mean_absolute_error(y_true=inputs, y_pred=decoded)
        self.assertLessEqual(error, 0.062)

        self.assertEqual(widths, group_widths)

    def test_encode_decode_large_padded(self):
        # Load the data
        with h5py.File('../../datasets/uci_har/train/data.h5', 'r') as fin:
            inputs = fin['inputs'][495]  # [50, 6]

        width = 8
        seq_length = inputs.shape[0]
        collected_indices = list(range(seq_length))
        non_fractional = 3

        flattened = inputs.T.reshape(-1)

        # Set the shifts
        shifts = select_range_shifts_array(measurements=flattened,
                                           old_width=16,
                                           old_precision=13,
                                           new_width=width,
                                           num_range_bits=3)

        merged_shifts, sizes = merge_shift_groups(values=flattened,
                                                  shifts=shifts,
                                                  max_num_groups=6)

        # Set the widths using the number of groups
        group_widths = [width for _ in sizes]

        # Encode and Decode the message
        encoded = message.encode_stable_measurements(measurements=inputs,
                                                     collected_indices=collected_indices,
                                                     seq_length=seq_length,
                                                     widths=group_widths,
                                                     group_sizes=sizes,
                                                     shifts=merged_shifts,
                                                     non_fractional=non_fractional)

        encoded = pad_to_length(encoded, length=len(encoded) + 12)

        decoded, indices, widths = message.decode_stable_measurements(encoded=encoded,
                                                                      seq_length=seq_length,
                                                                      num_features=inputs.shape[1],
                                                                      non_fractional=non_fractional)

        error = mean_absolute_error(y_true=inputs, y_pred=decoded)
        self.assertLessEqual(error, 0.01)

        self.assertEqual(widths, group_widths)


class TestDeltaEncode(unittest.TestCase):

    def test_encode(self):
        measurements = np.array([[10.0, 10.0], [12.0, 12.0], [12.5, 11.5]])
        
        encoded = message.delta_encode(measurements)
        expected = np.array([[10.0, 10.0], [2.0, 2.0], [0.5, -0.5]]) 

        self.assertTrue(np.all(np.isclose(encoded, expected)))

    def test_decode(self):
        encoded = np.array([[10.0, 10.0], [2.0, 2.0], [0.5, -0.5]]) 
        recovered = message.delta_decode(encoded)

        expected = np.array([[10.0, 10.0], [12.0, 12.0], [12.5, 11.5]])

        self.assertTrue(np.all(np.isclose(recovered, expected)))

    def test_single_feature(self):
        rand = np.random.RandomState(seed=3489)
        seq_length = 7

        measurements = rand.uniform(low=-2.0, high=2.0, size=(seq_length, 1))

        encoded = message.delta_encode(measurements)
        recovered = message.delta_decode(encoded)

        self.assertTrue(np.all(np.isclose(recovered, measurements)))

    def test_many_features(self):
        rand = np.random.RandomState(seed=3489)
        seq_length = 12
        num_features = 5

        measurements = rand.uniform(low=-2.0, high=2.0, size=(seq_length, num_features))

        encoded = message.delta_encode(measurements)
        recovered = message.delta_decode(encoded)

        self.assertTrue(np.all(np.isclose(recovered, measurements)))


if __name__ == '__main__':
    unittest.main()
