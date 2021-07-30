import unittest
import numpy as np
import h5py
from Cryptodome.Random import get_random_bytes

from adaptiveleak.energy_systems import get_group_target_bytes, EnergyUnit, convert_rate_to_energy
from adaptiveleak.utils import data_utils
from adaptiveleak.utils.constants import LENGTH_SIZE
from adaptiveleak.utils.encryption import AES_BLOCK_SIZE, encrypt_aes128, encrypt
from adaptiveleak.utils.data_types import EncryptionMode, CollectMode, PolicyType, EncodingMode
from adaptiveleak.utils.message import encode_standard_measurements


class TestQuantization(unittest.TestCase):

    def test_to_fp_pos(self):
        self.assertEqual(64, data_utils.to_fixed_point(0.25, precision=8, width=10))
        self.assertEqual(256, data_utils.to_fixed_point(0.25, precision=10, width=12))

        self.assertEqual(294, data_utils.to_fixed_point(0.28700833, precision=10, width=12))
        self.assertEqual(974, data_utils.to_fixed_point(0.95151288, precision=10, width=12))
        self.assertEqual(645, data_utils.to_fixed_point(0.63029945, precision=10, width=12))

        self.assertEqual(5, data_utils.to_fixed_point(0.28700833, precision=4, width=6))
        self.assertEqual(61, data_utils.to_fixed_point(0.95151288, precision=6, width=10))
        self.assertEqual(161, data_utils.to_fixed_point(0.63029945, precision=8, width=15))

    def test_to_fp_neg(self):
        self.assertEqual(-64, data_utils.to_fixed_point(-0.25, precision=8, width=10))
        self.assertEqual(-256, data_utils.to_fixed_point(-0.25, precision=10, width=12))

        self.assertEqual(-294, data_utils.to_fixed_point(-0.28700833, precision=10, width=12))
        self.assertEqual(-974, data_utils.to_fixed_point(-0.95151288, precision=10, width=12))
        self.assertEqual(-645, data_utils.to_fixed_point(-0.63029945, precision=10, width=12))

        self.assertEqual(-5, data_utils.to_fixed_point(-0.28700833, precision=4, width=6))
        self.assertEqual(-61, data_utils.to_fixed_point(-0.95151288, precision=6, width=10))
        self.assertEqual(-161, data_utils.to_fixed_point(-0.63029945, precision=8, width=15))

    def test_to_fp_range(self):
        self.assertEqual(511, data_utils.to_fixed_point(2, precision=8, width=10))
        self.assertEqual(511, data_utils.to_fixed_point(7, precision=8, width=10))

        self.assertEqual(255, data_utils.to_fixed_point(5, precision=6, width=9))
        self.assertEqual(255, data_utils.to_fixed_point(12, precision=6, width=9))

        self.assertEqual(-511, data_utils.to_fixed_point(-2, precision=8, width=10))
        self.assertEqual(-511, data_utils.to_fixed_point(-7, precision=8, width=10))

        self.assertEqual(-255, data_utils.to_fixed_point(-5, precision=6, width=9))
        self.assertEqual(-255, data_utils.to_fixed_point(-12, precision=6, width=9))

    def test_to_fp_neg_shift(self):
        self.assertEqual(1, data_utils.to_fixed_point(2, precision=-1, width=6))
        self.assertEqual(1, data_utils.to_fixed_point(4, precision=-2, width=7))
        self.assertEqual(2, data_utils.to_fixed_point(5, precision=-1, width=5))
        self.assertEqual(3, data_utils.to_fixed_point(12, precision=-2, width=3))
        self.assertEqual(3, data_utils.to_fixed_point(15, precision=-2, width=3))
        
        self.assertEqual(-1, data_utils.to_fixed_point(-2, precision=-1, width=6))
        self.assertEqual(-1, data_utils.to_fixed_point(-4, precision=-2, width=7))
        self.assertEqual(-2, data_utils.to_fixed_point(-5, precision=-1, width=5))
        self.assertEqual(-3, data_utils.to_fixed_point(-12, precision=-2, width=3))
        self.assertEqual(-3, data_utils.to_fixed_point(-15, precision=-2, width=3))

    def test_array_to_fp(self):
        array = np.array([0.25, -0.28700833, 0.95151288, 0.63029945])
        result = data_utils.array_to_fp(array, precision=10, width=12)
        expected = [256, -294, 974, 645]
        self.assertEqual(expected, result.tolist())

    def test_array_to_fp_shifted(self):
        array = np.array([0.25, -1.28700833, 0.95151288, 0.63029945])
        shifts = np.array([0, -1, -2, -2])

        result = data_utils.array_to_fp_shifted(array, precision=3, width=6, shifts=shifts)
        expected = [2, -21, 30, 20]

        self.assertEqual(expected, result.tolist())

    def test_to_float_pos(self):
        self.assertTrue(np.isclose(0.25, data_utils.to_float(64, precision=8)))
        self.assertTrue(np.isclose(0.25, data_utils.to_float(256, precision=10)))

        self.assertTrue(np.isclose(0.2861328125, data_utils.to_float(293, precision=10)))
        self.assertTrue(np.isclose(0.951171875, data_utils.to_float(974, precision=10)))
        self.assertTrue(np.isclose(0.6298828125, data_utils.to_float(645, precision=10)))

        self.assertTrue(np.isclose(0.25, data_utils.to_float(4, precision=4)))
        self.assertTrue(np.isclose(0.9375, data_utils.to_float(60, precision=6)))
        self.assertTrue(np.isclose(0.62890625, data_utils.to_float(161, precision=8)))

    def test_to_float_neg(self):
        self.assertTrue(np.isclose(-0.25, data_utils.to_float(-64, precision=8)))
        self.assertTrue(np.isclose(-0.25, data_utils.to_float(-256, precision=10)))

        self.assertTrue(np.isclose(-0.2861328125, data_utils.to_float(-293, precision=10)))
        self.assertTrue(np.isclose(-0.951171875, data_utils.to_float(-974, precision=10)))
        self.assertTrue(np.isclose(-0.6298828125, data_utils.to_float(-645, precision=10)))

        self.assertTrue(np.isclose(-0.25, data_utils.to_float(-4, precision=4)))
        self.assertTrue(np.isclose(-0.9375, data_utils.to_float(-60, precision=6)))
        self.assertTrue(np.isclose(-0.62890625, data_utils.to_float(-161, precision=8)))

    def test_to_float_neg_shift(self):
        self.assertEqual(2, data_utils.to_float(1, precision=-1))
        self.assertEqual(4, data_utils.to_float(1, precision=-2))
        self.assertEqual(4, data_utils.to_float(2, precision=-1))
        self.assertEqual(12, data_utils.to_float(3, precision=-2))
 
        self.assertEqual(-2, data_utils.to_float(-1, precision=-1))
        self.assertEqual(-4, data_utils.to_float(-1, precision=-2))
        self.assertEqual(-4, data_utils.to_float(-2, precision=-1))
        self.assertEqual(-12, data_utils.to_float(-3, precision=-2))

    def test_array_to_float(self):
        array = np.array([-256, 293, 974, -645])
        result = data_utils.array_to_float(array, precision=10)
        expected = np.array([-0.25, 0.2861328125, 0.951171875, -0.6298828125])
        self.assertTrue(np.all(np.isclose(expected, result)))

    def test_array_to_float_shifted(self):
        array = [-2, 21, 30, -20]
        shifts = np.array([0, -1, -2, -2])

        result = data_utils.array_to_float_shifted(array, precision=3, shifts=shifts)

        expected = [-0.25, 1.3125, 0.9375, -0.625]
        self.assertEqual(expected, result.tolist())

    def test_unsigned(self):
        array = np.array([10, 255, 12, 17, 0, 256])
        
        quantized = data_utils.array_to_fp(array, width=8, precision=0)
        recovered = data_utils.array_to_float(array, precision=0)

        self.assertTrue(np.all(np.isclose(array, recovered)))

    def test_neg_end_to_end(self):
        array = np.array([16, 32, 33, 48, 1024, 2047])

        quantized = data_utils.array_to_fp(array, width=8, precision=-4)
        recovered = data_utils.array_to_float(quantized, precision=-4)

        expected_quantized = np.array([1, 2, 2, 3, 64, 127])
        self.assertTrue(np.all(np.isclose(quantized, expected_quantized)))

        expected = np.array([16, 32, 32, 48, 1024, 2032])
        self.assertTrue(np.all(np.isclose(recovered, expected)))

    def test_end_to_end_eq_precision(self):
        value = -0.03125  # -1 / 32
        width = 4
        precision = 4

        fixed_point = data_utils.to_fixed_point(value, width=width, precision=precision)
        quantized = data_utils.to_float(fixed_point, precision=precision)

        self.assertTrue(np.isclose(quantized, 0.0))

    def test_end_to_end_higher_precision_lower(self):
        value = -0.03125  # -1 / 32
        width = 4
        precision = 5

        fixed_point = data_utils.to_fixed_point(value, width=width, precision=precision)
        quantized = data_utils.to_float(fixed_point, precision=precision)

        self.assertTrue(np.isclose(quantized, value))

    def test_end_to_end_higher_precision_upper(self):
        value = -0.25
        width = 4
        precision = 5

        fixed_point = data_utils.to_fixed_point(value, width=width, precision=precision)
        quantized = data_utils.to_float(fixed_point, precision=precision)

        self.assertTrue(np.isclose(quantized, -0.21875))


class TestRangeShift(unittest.TestCase):

    def test_range_single_int(self):
        old_width = 16
        old_precision = 7
        non_fractional = old_width - old_precision

        new_width = 8
        num_range_bits = 3

        value = (1 << (old_precision + 1))
        shift = data_utils.select_range_shift(measurement=value,
                                              old_width=old_width,
                                              old_precision=old_precision,
                                              new_width=new_width,
                                              num_range_bits=num_range_bits,
                                              prev_shift=1)
        self.assertEqual(shift, -4)

        float_value = 2.0

        new_precision = (new_width - non_fractional) - shift
        quantized = data_utils.to_fixed_point(float_value, width=new_width, precision=new_precision)
        recovered = data_utils.to_float(quantized, precision=new_precision)

        self.assertEqual(recovered, float_value)

    def test_range_single_int_2(self):
        old_width = 16
        old_precision = 0
        non_fractional = old_width - old_precision

        new_width = 6
        num_range_bits = 4

        value = data_utils.to_fixed_point(93, width=old_width, precision=old_precision)
        shift = data_utils.select_range_shift(measurement=value,
                                              old_width=old_width,
                                              old_precision=old_precision,
                                              new_width=new_width,
                                              num_range_bits=num_range_bits,
                                              prev_shift=1)
        self.assertEqual(shift, -4)

        float_value = 93.0

        new_precision = (new_width - non_fractional) - shift
        quantized = data_utils.to_fixed_point(float_value, width=new_width, precision=new_precision)
        recovered = data_utils.to_float(quantized, precision=new_precision)

        self.assertEqual(recovered, float_value)

    def test_range_single_float(self):
        old_width = 16
        old_precision = 13
        non_fractional = old_width - old_precision

        new_width = 14
        num_range_bits = 3

        float_value = 2.7236943
        value = data_utils.to_fixed_point(float_value, width=old_width, precision=old_precision)

        shift = data_utils.select_range_shift(measurement=value,
                                              old_width=old_width,
                                              old_precision=old_precision,
                                              new_width=new_width,
                                              num_range_bits=num_range_bits,
                                              prev_shift=-2)
        self.assertEqual(shift, 0)

        new_precision = (new_width - non_fractional) - shift
        quantized = data_utils.to_fixed_point(float_value, width=new_width, precision=new_precision)
        recovered = data_utils.to_float(quantized, precision=new_precision)

        self.assertLess(abs(recovered - float_value), 1e-4)

    def test_range_single_large(self):
        old_width = 16
        old_precision = 7
        non_fractional = old_width - old_precision

        new_width = 8
        num_range_bits = 3

        value = 0x4080

        shift = data_utils.select_range_shift(measurement=value,
                                              old_width=old_width,
                                              old_precision=old_precision,
                                              new_width=new_width,
                                              num_range_bits=num_range_bits,
                                              prev_shift=1)
        self.assertEqual(shift, 1)

        float_value = 129.0

        new_precision = (new_width - non_fractional) - shift
        quantized = data_utils.to_fixed_point(float_value, width=new_width, precision=new_precision)
        recovered = data_utils.to_float(quantized, precision=new_precision)

        self.assertEqual(recovered, 128.0)
    
    def test_range_single_large_2(self):
        old_width = 20
        old_precision = 8
        non_fractional = old_width - old_precision

        new_width = 13
        num_range_bits = 3

        float_value = -426.35
        value = data_utils.to_fixed_point(float_value, width=old_width, precision=old_precision)

        shift = data_utils.select_range_shift(measurement=value,
                                              old_width=old_width,
                                              old_precision=old_precision,
                                              new_width=new_width,
                                              num_range_bits=num_range_bits,
                                              prev_shift=1)
        self.assertEqual(shift, -2)

        new_precision = (new_width - non_fractional) - shift
        quantized = data_utils.to_fixed_point(float_value, width=new_width, precision=new_precision)
        recovered = data_utils.to_float(quantized, precision=new_precision)

        self.assertEqual(recovered, -426.375)

    def test_range_single_large_exact(self):
        old_width = 16
        old_precision = 7
        non_fractional = old_width - old_precision

        new_width = 8
        num_range_bits = 3

        value = 0x4100

        shift = data_utils.select_range_shift(measurement=value,
                                              old_width=old_width,
                                              old_precision=old_precision,
                                              new_width=new_width,
                                              num_range_bits=num_range_bits,
                                              prev_shift=1)
        self.assertEqual(shift, 0)

        float_value = data_utils.to_float(value, precision=old_precision)

        new_precision = (new_width - non_fractional) - shift
        quantized = data_utils.to_fixed_point(float_value, width=new_width, precision=new_precision)
        recovered = data_utils.to_float(quantized, precision=new_precision)

        self.assertEqual(recovered, float_value)

    def test_range_single_frac(self):
        old_width = 16
        old_precision = 7
        non_fractional = old_width - old_precision

        new_width = 8
        num_range_bits = 3

        value = 0x0011

        shift = data_utils.select_range_shift(measurement=value,
                                              old_width=old_width,
                                              old_precision=old_precision,
                                              new_width=new_width,
                                              num_range_bits=num_range_bits,
                                              prev_shift=1)
        self.assertEqual(shift, -4)

        float_value = data_utils.to_float(value, precision=old_precision)

        new_precision = (new_width - non_fractional) - shift
        quantized = data_utils.to_fixed_point(float_value, width=new_width, precision=new_precision)
        recovered = data_utils.to_float(quantized, precision=new_precision)

        self.assertEqual(recovered, 0.125)

    def test_range_arr_integers(self):
        measurements = np.array([1.0, 2.0, -3.0, 4.0, -1.0, 3.0])
        old_width = 16
        old_precision = 7
        non_fractional = old_width - old_precision

        new_width = 8
        num_range_bits = 3

        shifts = data_utils.select_range_shifts_array(measurements=measurements,
                                                      old_width=old_width,
                                                      old_precision=old_precision,
                                                      new_width=new_width,
                                                      num_range_bits=num_range_bits)
        shifts_list = shifts.tolist()

        self.assertEqual(shifts_list, [-4, -4, -4, -4, -4, -4])

        new_precision = new_width - non_fractional

        quantized = data_utils.array_to_fp_shifted(measurements, width=new_width, precision=new_precision, shifts=shifts)
        recovered = data_utils.array_to_float_shifted(quantized, precision=new_precision, shifts=shifts)

        recovered_list = recovered.tolist()

        self.assertEqual(recovered_list, [1.0, 2.0, -3.0, 4.0, -1.0, 3.0])

    def test_range_arr_integers_2(self):
        measurements = np.array([0, 76, 153, 229, 306, 382, 23, 11, 11, 11, 11, 0, 79, 159, 238, 318, 398, 415, 455])
        old_width = 16
        old_precision = 0
        non_fractional = old_width - old_precision

        new_width = 8
        num_range_bits = 4

        shifts = data_utils.select_range_shifts_array(measurements=measurements,
                                                      old_width=old_width,
                                                      old_precision=old_precision,
                                                      new_width=new_width,
                                                      num_range_bits=num_range_bits)
        shifts_list = shifts.tolist()
        print(shifts_list)

    def test_range_arr_mixed_one(self):
        measurements = np.array([1.75, 2.0, -3.5, 4.75, -1.0, 0.1875])
        old_width = 8
        old_precision = 4
        non_fractional = old_width - old_precision

        new_width = 5
        num_range_bits = 3

        shifts = data_utils.select_range_shifts_array(measurements=measurements,
                                                      old_width=old_width,
                                                      old_precision=old_precision,
                                                      new_width=new_width,
                                                      num_range_bits=num_range_bits)
        shifts_list = shifts.tolist()
        
        self.assertEqual(shifts_list, [-2, -1, -1, 0, 0, -4])

        new_precision = new_width - non_fractional
        
        quantized = data_utils.array_to_fp_shifted(measurements, width=new_width, precision=new_precision, shifts=shifts)
        recovered = data_utils.array_to_float_shifted(quantized, precision=new_precision, shifts=shifts)

        recovered_list = recovered.tolist()
        
        self.assertEqual(recovered_list, [1.75, 2.0, -3.5, 5.0, -1.0, 0.1875])

    def test_range_arr_mixed_two(self):
        measurements = np.array([1.5, 2.05, -3.5, 1.03125])
        old_width = 10
        old_precision = 7
        non_fractional = old_width - old_precision

        new_width = 7
        num_range_bits = 4

        shifts = data_utils.select_range_shifts_array(measurements=measurements,
                                                      old_width=old_width,
                                                      old_precision=old_precision,
                                                      new_width=new_width,
                                                      num_range_bits=num_range_bits)
        shifts_list = shifts.tolist()

        self.assertEqual(shifts_list, [0, 0, 0, -1])

        new_precision = new_width - non_fractional
        
        quantized = data_utils.array_to_fp_shifted(measurements, width=new_width, precision=new_precision, shifts=shifts)
        recovered = data_utils.array_to_float_shifted(quantized, precision=new_precision, shifts=shifts)

        recovered_list = recovered.tolist()
        
        self.assertEqual(recovered_list, [1.5, 2.0625, -3.5, 1.03125])

    def test_range_arr_border(self):
        measurements = np.array([1.75, 1.0])

        old_width = 6
        old_precision = 4
        non_fractional = old_width - old_precision

        new_width = 4
        num_range_bits = 2

        shifts = data_utils.select_range_shifts_array(measurements=measurements,
                                                      old_width=old_width,
                                                      old_precision=old_precision,
                                                      new_width=new_width,
                                                      num_range_bits=num_range_bits)
        shifts_list = shifts.tolist()
        self.assertEqual(shifts_list, [0, 0])

        new_precision = new_width - non_fractional
        
        quantized = data_utils.array_to_fp_shifted(measurements, width=new_width, precision=new_precision, shifts=shifts)
        recovered = data_utils.array_to_float_shifted(quantized, precision=new_precision, shifts=shifts)

        recovered_list = recovered.tolist()
        
        self.assertEqual(recovered_list, [1.75, 1.0])

    def test_range_larger_integers(self):
        measurements = np.array([65.0, 63.0, 64.0, 8192.0, 9216.0, 8192.0])
       
        old_width = 16
        old_precision = 0
        non_fractional = old_width - old_precision
        
        new_width = 13
        num_range_bits = 3

        shifts = data_utils.select_range_shifts_array(measurements=measurements,
                                                      old_width=old_width,
                                                      old_precision=old_precision,
                                                      new_width=new_width,
                                                      num_range_bits=num_range_bits)

        shifts_list = shifts.tolist()
        self.assertEqual(shifts_list, [-4, -4, -4, -1, -1, -1])

        new_precision = new_width - non_fractional

        quantized = data_utils.array_to_fp_shifted(measurements, width=new_width, precision=new_precision, shifts=shifts)
        recovered = data_utils.array_to_float_shifted(quantized, precision=new_precision, shifts=shifts)

        recovered_list = recovered.tolist()
        
        self.assertEqual(recovered_list, measurements.tolist())


class TestExtrapolation(unittest.TestCase):

    def test_one(self):
        prev = np.array([1, 1, 1, 1], dtype=float)
        curr = np.array([2, 2, 2, 2], dtype=float)

        predicted = data_utils.linear_extrapolate(prev=prev, curr=curr, delta=1, num_steps=1)
        expected = np.array([3, 3, 3, 3], dtype=float)

        self.assertTrue(np.all(np.isclose(predicted, expected)))

    def test_rand(self):
        size = 10

        rand = np.random.RandomState(seed=38)
        m = rand.uniform(low=-1.0, high=1.0, size=size)
        b = rand.uniform(low=-1.0, high=1.0, size=size)

        t0 = np.ones_like(m) * 1.0
        prev = m * t0 + b

        t1 = np.ones_like(m) * 1.25
        curr = m * t1 + b

        predicted = data_utils.linear_extrapolate(prev=prev, curr=curr, delta=0.25, num_steps=1)
        
        t2 = np.ones_like(m) * 1.5
        expected = m * t2 + b

        self.assertTrue(np.all(np.isclose(predicted, expected)))


class TestPadding(unittest.TestCase):

    def test_round_is_multiple(self):
        message = get_random_bytes(AES_BLOCK_SIZE)
        key = get_random_bytes(AES_BLOCK_SIZE)

        padded_size = data_utils.round_to_block(length=len(message), block_size=AES_BLOCK_SIZE)
        padded_size += AES_BLOCK_SIZE  # Account for the IV

        expected_size = len(encrypt_aes128(message, key=key))

        self.assertEqual(padded_size, expected_size)

    def test_round_non_multiple(self):
        message = get_random_bytes(9)
        key = get_random_bytes(AES_BLOCK_SIZE)

        padded_size = data_utils.round_to_block(length=len(message), block_size=AES_BLOCK_SIZE)
        padded_size += AES_BLOCK_SIZE  # Account for the IV

        expected_size = len(encrypt_aes128(message, key=key))

        self.assertEqual(padded_size, expected_size)

    def test_pad_below(self):
        message = get_random_bytes(9)
        padded = data_utils.pad_to_length(message=message, length=AES_BLOCK_SIZE)

        self.assertEqual(len(padded), AES_BLOCK_SIZE)

    def test_pad_above(self):
        message = get_random_bytes(20)
        padded = data_utils.pad_to_length(message=message, length=AES_BLOCK_SIZE)

        self.assertEqual(len(padded), 20)

    def test_pad_equal(self):
        message = get_random_bytes(AES_BLOCK_SIZE)
        padded = data_utils.pad_to_length(message=message, length=AES_BLOCK_SIZE)

        self.assertEqual(len(padded), AES_BLOCK_SIZE)


class TestPacking(unittest.TestCase):

    def test_pack_single_byte_ones(self):
        values = [0xF, 0xF]
        packed = data_utils.pack(values, width=4)

        expected = bytes([0xFF])

        self.assertEqual(packed, expected)

    def test_pack_single_byte_diff(self):
        values = [0x1, 0xF]
        packed = data_utils.pack(values, width=4)

        expected = bytes([0xF1])

        self.assertEqual(packed, expected)

    def test_pack_full_byte(self):
        values = [0xAB, 0xCD]
        packed = data_utils.pack(values, width=8)

        expected = bytes([0xAB, 0xCD])
        self.assertEqual(packed, expected)

    def test_pack_multi_byte(self):
        values = [0x1, 0x12]
        packed = data_utils.pack(values, width=5)

        expected = bytes([0x41, 0x2])

        self.assertEqual(packed, expected)

    def test_pack_multi_byte_three(self):
        values = [0x1, 0x12, 0x06]
        packed = data_utils.pack(values, width=5)

        expected = bytes([0x41, 0x1A])

        self.assertEqual(packed, expected)

    def test_pack_multi_byte_values(self):
        values = [0x101, 0x092]
        packed = data_utils.pack(values, width=9)

        expected = bytes([0x01, 0x25, 0x01])

        self.assertEqual(packed, expected)

    def test_unpack_single_byte_ones(self):
        encoded = bytes([0xFF])
        values = data_utils.unpack(encoded, width=4, num_values=2)

        self.assertEqual(len(values), 2)
        self.assertEqual(values[0], 0xF)
        self.assertEqual(values[1], 0xF)

    def test_unpack_single_byte_diff(self):
        encoded = bytes([0xF1])
        values = data_utils.unpack(encoded, width=4, num_values=2)

        self.assertEqual(len(values), 2)
        self.assertEqual(values[0], 0x1)
        self.assertEqual(values[1], 0xF)

    def test_unpack_full_byte(self):
        encoded = bytes([0xAB, 0xCD])
        values = data_utils.unpack(encoded, width=8, num_values=2)

        expected = [0xAB, 0xCD]
        self.assertEqual(values, expected)

    def test_unpack_multi_byte(self):
        encoded = bytes([0x41, 0x2])
        values = data_utils.unpack(encoded, width=5, num_values=2)

        self.assertEqual(len(values), 2)
        self.assertEqual(values[0], 0x1)
        self.assertEqual(values[1], 0x12)

    def test_unpack_multi_byte_three(self):
        encoded = bytes([0x41, 0x1A])
        values = data_utils.unpack(encoded, width=5, num_values=3)

        self.assertEqual(len(values), 3)
        self.assertEqual(values[0], 0x1)
        self.assertEqual(values[1], 0x12)
        self.assertEqual(values[2], 0x06)

    def test_unpack_multi_byte_values(self):
        encoded = bytes([0x01, 0x25, 0x01])
        values = data_utils.unpack(encoded, width=9, num_values=2)

        self.assertEqual(len(values), 2)
        self.assertEqual(values[0], 0x101)
        self.assertEqual(values[1], 0x092)


class TestGroupTargetBytes(unittest.TestCase):

    def test_target_block(self):
        encryption_mode = EncryptionMode.BLOCK
        width = 7
        num_features = 3
        seq_length = 10
        period = 10
        rate = 0.2

        energy_unit = EnergyUnit(policy_type=PolicyType.ADAPTIVE_HEURISTIC,
                                 encoding_mode=EncodingMode.GROUP,
                                 encryption_mode=encryption_mode,
                                 collect_mode=CollectMode.LOW,
                                 seq_length=seq_length,
                                 num_features=num_features,
                                 period=period)

        target_energy = convert_rate_to_energy(collection_rate=rate,
                                               width=width,
                                               encryption_mode=encryption_mode,
                                               collect_mode=CollectMode.LOW,
                                               seq_length=seq_length,
                                               num_features=num_features)

        target_bytes = get_group_target_bytes(width=width,
                                              collection_rate=rate,
                                              num_features=num_features,
                                              seq_length=seq_length,
                                              encryption_mode=encryption_mode,
                                              energy_unit=energy_unit,
                                              target_energy=target_energy)

        self.assertEqual(target_bytes, 16)

    def test_target_stream(self):
        encryption_mode = EncryptionMode.STREAM
        width = 7
        num_features = 3
        seq_length = 10
        period = 10
        rate = 0.2

        energy_unit = EnergyUnit(policy_type=PolicyType.ADAPTIVE_HEURISTIC,
                                 encoding_mode=EncodingMode.GROUP,
                                 encryption_mode=encryption_mode,
                                 collect_mode=CollectMode.LOW,
                                 seq_length=seq_length,
                                 num_features=num_features,
                                 period=period)

        target_energy = convert_rate_to_energy(collection_rate=rate,
                                               width=width,
                                               encryption_mode=encryption_mode,
                                               collect_mode=CollectMode.LOW,
                                               seq_length=seq_length,
                                               num_features=num_features)

        target_bytes = get_group_target_bytes(width=width,
                                              collection_rate=rate,
                                              num_features=num_features,
                                              seq_length=seq_length,
                                              encryption_mode=encryption_mode,
                                              energy_unit=energy_unit,
                                              target_energy=target_energy)

        self.assertEqual(target_bytes, 19)

    def test_target_stream_large(self):
        encryption_mode = EncryptionMode.STREAM
        width = 16
        num_features = 1
        seq_length = 1250
        period = 10
        rate = 0.7

        energy_unit = EnergyUnit(policy_type=PolicyType.ADAPTIVE_HEURISTIC,
                                 encoding_mode=EncodingMode.GROUP,
                                 encryption_mode=encryption_mode,
                                 collect_mode=CollectMode.LOW,
                                 seq_length=seq_length,
                                 num_features=num_features,
                                 period=period)

        target_energy = convert_rate_to_energy(collection_rate=rate,
                                               width=width,
                                               encryption_mode=encryption_mode,
                                               collect_mode=CollectMode.LOW,
                                               seq_length=seq_length,
                                               num_features=num_features)

        target_bytes = get_group_target_bytes(width=width,
                                              collection_rate=rate,
                                              num_features=num_features,
                                              seq_length=seq_length,
                                              encryption_mode=encryption_mode,
                                              energy_unit=energy_unit,
                                              target_energy=target_energy)

        self.assertEqual(target_bytes, 1759)


class TestCalculateBytes(unittest.TestCase):

    def test_byte_block(self):
        # 42 bits -> 6 bytes of data, 2 for sequence mask, 2 for length, 16 for IV = 26 bytes -> 32 bytes
        data_bytes = data_utils.calculate_bytes(width=7,
                                                num_features=3,
                                                num_collected=2,
                                                encryption_mode=EncryptionMode.BLOCK,
                                                seq_length=10)
        self.assertEqual(data_bytes, 34)

    def test_byte_stream(self):
        # 42 bits -> 6 bytes of data, 2 for sequence mask, 2 for length, 12 for nonce = 22 bytes
        data_bytes = data_utils.calculate_bytes(width=7,
                                                num_features=3,
                                                num_collected=2,
                                                seq_length=9,
                                                encryption_mode=EncryptionMode.STREAM)

        self.assertEqual(data_bytes, 22)

    def test_byte_stream_end_to_end_one(self):
        # Encode and encrypt measurements
        measurements = np.ones(shape=(2, 3))
        collected_indices = [0, 6]
        seq_length = 8
        precision = 4
        width = 8

        encoded = encode_standard_measurements(measurements=measurements,
                                              collected_indices=collected_indices,
                                              seq_length=seq_length,
                                              precision=precision,
                                              width=width,
                                              should_compress=False)

        key = get_random_bytes(32)
        encrypted = encrypt(message=encoded, key=key, mode=EncryptionMode.STREAM)

        message_bytes = data_utils.calculate_bytes(width=width,
                                                   num_features=measurements.shape[1],
                                                   num_collected=measurements.shape[0],
                                                   seq_length=seq_length,
                                                   encryption_mode=EncryptionMode.STREAM)

        self.assertEqual(message_bytes, len(encrypted) + LENGTH_SIZE)

    def test_byte_stream_end_to_end_two(self):
        # Encode and encrypt measurements
        measurements = np.ones(shape=(2, 3))
        collected_indices = [0, 6]
        seq_length = 12
        precision = 4
        width = 12

        encoded = encode_standard_measurements(measurements=measurements,
                                               collected_indices=collected_indices,
                                               seq_length=seq_length,
                                               precision=precision,
                                               width=width,
                                               should_compress=False)

        key = get_random_bytes(32)
        encrypted = encrypt(message=encoded, key=key, mode=EncryptionMode.STREAM)

        message_bytes = data_utils.calculate_bytes(width=width,
                                                   num_features=measurements.shape[1],
                                                   num_collected=measurements.shape[0],
                                                   seq_length=seq_length,
                                                   encryption_mode=EncryptionMode.STREAM)

        self.assertEqual(message_bytes, len(encrypted) + LENGTH_SIZE)

    def test_byte_block_end_to_end_one(self):
        # Encode and encrypt measurements
        measurements = np.ones(shape=(2, 3))
        collected_indices = [0, 6]
        seq_length = 8
        precision = 4
        width = 8

        encoded = encode_standard_measurements(measurements=measurements,
                                               collected_indices=collected_indices,
                                               seq_length=seq_length,
                                               precision=precision,
                                               width=width,
                                               should_compress=False)

        key = get_random_bytes(AES_BLOCK_SIZE)
        encrypted = encrypt(message=encoded, key=key, mode=EncryptionMode.BLOCK)

        message_bytes = data_utils.calculate_bytes(width=width,
                                                   num_features=measurements.shape[1],
                                                   num_collected=measurements.shape[0],
                                                   seq_length=seq_length,
                                                   encryption_mode=EncryptionMode.BLOCK)

        self.assertEqual(message_bytes, len(encrypted) + LENGTH_SIZE)

    def test_byte_block_end_to_end_two(self):
        # Encode and encrypt measurements
        measurements = np.ones(shape=(2, 3))
        collected_indices = [0, 6]
        seq_length = 9
        precision = 4
        width = 12

        encoded = encode_standard_measurements(measurements=measurements,
                                               collected_indices=collected_indices,
                                               seq_length=seq_length,
                                               precision=precision,
                                               width=width,
                                               should_compress=False)

        key = get_random_bytes(AES_BLOCK_SIZE)
        encrypted = encrypt(message=encoded, key=key, mode=EncryptionMode.BLOCK)

        message_bytes = data_utils.calculate_bytes(width=width,
                                                   num_features=measurements.shape[1],
                                                   num_collected=measurements.shape[0],
                                                   seq_length=seq_length,
                                                   encryption_mode=EncryptionMode.BLOCK)

        self.assertEqual(message_bytes, len(encrypted) + LENGTH_SIZE)

    def test_group_block(self):
        # 11 bytes of data, 3 meta-data, 2 for sequence mask = 16 bytes -> 32 bytes + 16 byte IV + 2 byte length = 50 bytes
        data_bytes = data_utils.calculate_grouped_bytes(widths=[6, 7],
                                                        num_features=3,
                                                        num_collected=4,
                                                        seq_length=10,
                                                        group_size=6,
                                                        encryption_mode=EncryptionMode.BLOCK)
        self.assertEqual(data_bytes, 50)

    def test_group_stream(self):
        # 11 bytes of data, 3 meta-data, 2 for sequence mask, 12 for nonce = 28 bytes
        data_bytes = data_utils.calculate_grouped_bytes(widths=[6, 7],
                                                        num_features=3,
                                                        num_collected=4,
                                                        seq_length=9,
                                                        group_size=6,
                                                        encryption_mode=EncryptionMode.STREAM)

        self.assertEqual(data_bytes, 30)

    def test_group_stream_unbalanced(self):
        # 11 bytes of data, 3 meta-data, 2 for sequence mask, 12 for nonce = 29 bytes
        data_bytes = data_utils.calculate_grouped_bytes(widths=[6, 7],
                                                        num_features=3,
                                                        num_collected=4,
                                                        seq_length=9,
                                                        group_size=6,
                                                        encryption_mode=EncryptionMode.STREAM)

        self.assertEqual(data_bytes, 30)

    def test_group_stream_large(self):
        data_bytes = data_utils.calculate_grouped_bytes(widths=[7, 9],
                                                        num_features=6,
                                                        num_collected=26,
                                                        seq_length=50,
                                                        group_size=132,
                                                        encryption_mode=EncryptionMode.STREAM)
        self.assertEqual(data_bytes, 167)


class TestGroupWidths(unittest.TestCase):

    def test_widths_block_above(self):
        widths = data_utils.get_group_widths(group_size=6,
                                             num_collected=6,
                                             num_features=3,
                                             seq_length=10,
                                             target_frac=0.5,
                                             standard_width=8,
                                             encryption_mode=EncryptionMode.BLOCK)

        self.assertEqual(len(widths), 3)
        self.assertEqual(widths[0], 11)
        self.assertEqual(widths[1], 10)
        self.assertEqual(widths[2], 10)

    def test_widths_block_below(self):
        widths = data_utils.get_group_widths(group_size=6,
                                             num_collected=3,
                                             num_features=3,
                                             seq_length=10,
                                             target_frac=0.5,
                                             standard_width=8,
                                             encryption_mode=EncryptionMode.BLOCK)

        self.assertEqual(len(widths), 2)
        self.assertEqual(widths[0], 22)
        self.assertEqual(widths[1], 22)

    def test_widths_stream_above(self):
        widths = data_utils.get_group_widths(group_size=6,
                                             num_collected=6,
                                             num_features=3,
                                             seq_length=10,
                                             target_frac=0.5,
                                             standard_width=8,
                                             encryption_mode=EncryptionMode.STREAM)

        self.assertEqual(len(widths), 3)
        self.assertEqual(widths[0], 5)
        self.assertEqual(widths[1], 5)
        self.assertEqual(widths[2], 5)

    def test_widths_stream_below(self):
        widths = data_utils.get_group_widths(group_size=6,
                                             num_collected=3,
                                             num_features=3,
                                             seq_length=10,
                                             target_frac=0.5,
                                             standard_width=8,
                                             encryption_mode=EncryptionMode.STREAM)

        self.assertEqual(len(widths), 2)
        self.assertEqual(widths[0], 10)
        self.assertEqual(widths[1], 10)


#class TestMaxGroups(unittest.TestCase):
#
#    def test_size_one_feature(self):
#        target_frac = 0.2
#        seq_length = 235
#        num_features = 1
#        min_width = 3
#        group_size = 100
#        standard_width = 8
#        encryption_mode = EncryptionMode.STREAM
#
#        target_collected = int(seq_length * target_frac)
#        target_size = data_utils.calculate_bytes(width=standard_width,
#                                                 num_collected=target_collected,
#                                                 num_features=num_features,
#                                                 seq_length=seq_length,
#                                                 encryption_mode=encryption_mode)
#
#        max_collected = data_utils.get_max_collected(seq_length=seq_length,
#                                                     num_features=num_features,
#                                                     group_size=group_size,
#                                                     min_width=min_width,
#                                                     target_size=target_size,
#                                                     encryption_mode=encryption_mode)
#
#        # Verify the count
#        self.assertEqual(max_collected, 113)
#
#        # Verify the number of bytes
#        grouped_size = data_utils.calculate_grouped_bytes(widths=[min_width, min_width],
#                                                          num_collected=max_collected,
#                                                          num_features=num_features,
#                                                          group_size=group_size,
#                                                          encryption_mode=encryption_mode,
#                                                          seq_length=seq_length)
#
#        self.assertLessEqual(grouped_size, target_size)
#
#    def test_size_multi_feature(self):
#        target_frac = 0.2
#        seq_length = 50
#        num_features = 8
#        min_width = 3
#        group_size = 128
#        encryption_mode = EncryptionMode.STREAM
#
#        target_collected = int(seq_length * target_frac)
#        target_size = data_utils.calculate_bytes(width=8,
#                                                 num_collected=target_collected,
#                                                 num_features=num_features,
#                                                 seq_length=seq_length,
#                                                 encryption_mode=encryption_mode)
#
#        max_collected = data_utils.get_max_collected(seq_length=seq_length,
#                                                     num_features=num_features,
#                                                     group_size=group_size,
#                                                     min_width=min_width,
#                                                     target_size=target_size,
#                                                     encryption_mode=encryption_mode)
#
#        # Verify the count
#        self.assertEqual(max_collected, 25)
#
#        # Verify the number of bytes
#        grouped_size = data_utils.calculate_grouped_bytes(widths=[min_width, min_width],
#                                                          num_collected=max_collected,
#                                                          num_features=num_features,
#                                                          group_size=group_size,
#                                                          encryption_mode=encryption_mode,
#                                                          seq_length=seq_length)
#
#        self.assertLessEqual(grouped_size, target_size)
#
#    def test_size_long_seq(self):
#        target_frac = 0.3
#        seq_length = 206
#        num_features = 3
#        min_width = 4
#        group_size = 119
#        encryption_mode = EncryptionMode.STREAM
#
#        target_collected = int(seq_length * target_frac)
#        target_size = data_utils.calculate_bytes(width=8,
#                                                 num_collected=target_collected,
#                                                 num_features=num_features,
#                                                 seq_length=seq_length,
#                                                 encryption_mode=encryption_mode)
#
#        max_collected = data_utils.get_max_collected(seq_length=seq_length,
#                                                     num_features=num_features,
#                                                     group_size=group_size,
#                                                     min_width=min_width,
#                                                     target_size=target_size,
#                                                     encryption_mode=encryption_mode)
#
#        # Verify the count
#        self.assertEqual(max_collected, 118)
#
#        # Verify the number of bytes
#        grouped_size = data_utils.calculate_grouped_bytes(widths=[min_width, min_width, min_width],
#                                                          num_collected=max_collected,
#                                                          num_features=num_features,
#                                                          group_size=group_size,
#                                                          encryption_mode=encryption_mode,
#                                                          seq_length=seq_length)
#
#        self.assertLessEqual(grouped_size, target_size)


class TestPruning(unittest.TestCase):

    def test_prune_two(self):
        measurements = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.5, 4.0], [3.5, 3.0]])
        max_collected = 3
        collected_indices = [1, 3, 5, 9, 10]
        seq_length = 12

        # Iter 0 -> Expected Diffs: [0, 2, 2.5, 2.0], Expected Diff Idx: [2, 4, 1, 2]  -> Prune 1
        # iter 1 -> Expected Diffs: [2, 2.5, 2.0], Expected Diff Idx: [4, 1, 2] -> Prune 3

        # Errors -> [1.0, 0.5, 1.9] -> Prune 1, 2

        pruned_features, pruned_indices = data_utils.prune_sequence(measurements=measurements,
                                                                    max_collected=max_collected,
                                                                    collected_indices=collected_indices,
                                                                    seq_length=seq_length)

        expected_features = np.array([[1.0, 1.0], [2.5, 4.0], [3.5, 3.0]])
        expected_indices = [1, 9, 10]

        self.assertTrue(np.all(np.isclose(pruned_features, expected_features)))
        self.assertEqual(pruned_indices, expected_indices)

    def test_prune_middle(self):
        measurements = np.array([[1.0, 1.0], [1.5, 1.5], [3.0, 3.0], [3.2, 3.0], [3.5, 3.0]])
        max_collected = 3
        collected_indices = [1, 3, 5, 7, 10]
        seq_length = 11

        # Iter 0 -> Expected Diffs: [0, 2, 2.5, 2.0], Expected Diff Idx: [2, 2, 3, 1]  -> Prune 1
        # iter 1 -> Expected Diffs: [2, 2.5, 2.0], Expected Diff Idx: [4, 3, 1] -> Prune 4

        # Errors: [1.0, 1.4, 0.0]

        pruned_features, pruned_indices = data_utils.prune_sequence(measurements=measurements,
                                                                    max_collected=max_collected,
                                                                    collected_indices=collected_indices,
                                                                    seq_length=seq_length)

        expected_features = np.array([[1.0, 1.0], [3.0, 3.0], [3.5, 3.0]])
        expected_indices = [1, 5, 10]

        self.assertTrue(np.all(np.isclose(pruned_features, expected_features)))
        self.assertEqual(pruned_indices, expected_indices)

    def test_prune_final_first(self):
        measurements = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        max_collected = 2
        collected_indices = [1, 5, 10, 15]
        seq_length = 16

        # Iter 0 -> Expected Diffs: [2, 2, 2], Expected Diff Idx: [5, 5, 1]  -> Prune 3
        # iter 1 -> Expected Diffs: [2, 2], Expected Diff Idx: [5, 5] -> Prune 1 (break ties first)

        pruned_features, pruned_indices = data_utils.prune_sequence(measurements=measurements,
                                                                    max_collected=max_collected,
                                                                    collected_indices=collected_indices,
                                                                    seq_length=seq_length)

        expected_features = np.array([[1.0, 1.0], [4.0, 4.0]])
        expected_indices = [1, 15]

        self.assertTrue(np.all(np.isclose(pruned_features, expected_features)))
        self.assertEqual(pruned_indices, expected_indices)


class TestRLE(unittest.TestCase):

    def test_rle_small(self):
        values = [1, 1, 1, 1, 1, 0, 0, 3, 3]
        signs = [1, 1, 0, 0, 0, 1, 0, 1, 1]

        encoded = data_utils.run_length_encode(values, signs)

        decoded_vals, decoded_signs = data_utils.run_length_decode(encoded)

        self.assertEqual(decoded_vals, values)
        self.assertEqual(decoded_signs, signs)

    def test_rle_small_two(self):
        values = [0, 0, 0, 0, 0, 0]
        signs = [1, 0, 0, 1, 1, 0]

        encoded = data_utils.run_length_encode(values, signs)
        decoded_vals, decoded_signs = data_utils.run_length_decode(encoded)

        self.assertEqual(decoded_vals, values)
        self.assertEqual(decoded_signs, signs)

    def test_rle_long(self):
        with h5py.File('../../datasets/uci_har/train/data.h5', 'r') as fin:
            inputs = fin['inputs'][0]

        flattened = inputs.reshape(-1)
        integer_parts = list(map(data_utils.integer_part, np.abs(flattened)))
        signs = data_utils.get_signs(flattened)

        encoded = data_utils.run_length_encode(integer_parts, signs)
        decoded_vals, decoded_signs = data_utils.run_length_decode(encoded)

        decoded = data_utils.apply_signs(decoded_vals, decoded_signs)

        self.assertTrue(decoded, flattened.tolist())


class TestPartExtraction(unittest.TestCase):

    def test_half(self):
        precision = 4
        width = 8
        fixed_point_value = data_utils.to_fixed_point(1.5, width=width, precision=precision)

        int_part = data_utils.fixed_point_integer_part(fixed_point_value, precision=precision)
        frac_part = data_utils.fixed_point_frac_part(fixed_point_value, precision=precision)

        self.assertTrue(frac_part, 8)
        self.assertTrue(int_part, 1)

    def test_neg_eighth(self):
        precision = 3
        width = 8
        fixed_point_value = data_utils.to_fixed_point(-1.125, width=width, precision=precision)

        int_part = data_utils.fixed_point_integer_part(fixed_point_value, precision=precision)
        frac_part = data_utils.fixed_point_frac_part(fixed_point_value, precision=precision)

        self.assertTrue(frac_part, 1)
        self.assertTrue(int_part, -1)


if __name__ == '__main__':
    unittest.main()
