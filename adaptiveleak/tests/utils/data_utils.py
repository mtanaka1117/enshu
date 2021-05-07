import unittest
import numpy as np
import h5py
from Cryptodome.Random import get_random_bytes

from adaptiveleak.utils import data_utils
from adaptiveleak.utils.encryption import AES_BLOCK_SIZE, encrypt_aes128, EncryptionMode, encrypt
from adaptiveleak.utils.message import encode_standard_measurements, encode_grouped_measurements


class TestNeuralNetwork(unittest.TestCase):

    def test_leaky_relu_quarter(self):
        array = np.array([-1.0, 1.0, 0.25, 0.0, -0.75, 100.0, -1000.0])
        result = data_utils.leaky_relu(array, alpha=0.25)
        expected = np.array([-0.25, 1.0, 0.25, 0.0, -0.1875, 100.0, -250.0])

        self.assertTrue(np.all(np.isclose(result, expected)))

    def test_leaky_relu_half(self):
        array = np.array([-3.0, -1.0, 0.14, -0.42, 0.0, -100.0, 1000.0])
        result = data_utils.leaky_relu(array, alpha=0.5)
        expected = np.array([-1.5, -0.5, 0.14, -0.21, 0.0, -50.0, 1000.0])

        self.assertTrue(np.all(np.isclose(result, expected)))

    def test_softmax_equal(self):
        array = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        result = data_utils.softmax(array, axis=-1)
        expected = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        self.assertTrue(np.all(np.isclose(result, expected)))

    def test_softmax_unequal(self):
        array = np.array([0.28700833, 0.95151288, 0.63029945, -0.61770699, 0.1945032, 0.49076853])
        result = data_utils.softmax(array, axis=-1)
        expected = np.array([0.14502396363014, 0.2818580414792, 0.20442274216539, 0.058684971692022, 0.13221030377376, 0.17779997725948])
        self.assertTrue(np.all(np.isclose(result, expected)))

    def test_softmax_last_axis(self):
        array = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.28700833, 0.95151288, 0.63029945, -0.61770699, 0.1945032, 0.49076853]])
        result = data_utils.softmax(array, axis=-1)

        one_sixth = 1.0 / 6.0
        expected = np.array([[one_sixth, one_sixth, one_sixth, one_sixth, one_sixth, one_sixth], [0.14502396363014, 0.2818580414792, 0.20442274216539, 0.058684971692022, 0.13221030377376, 0.17779997725948]])

        self.assertTrue(np.all(np.isclose(result, expected)))
        
    def test_softmax_first_axis(self):
        array = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.28700833, 0.95151288, 0.63029945, -0.61770699, 0.1945032, 0.49076853]])
        result = data_utils.softmax(array, axis=0)

        first_row = [0.55304752522136, 0.38900112554213, 0.46747114724301, 0.75356313896948, 0.57578570225175, 0.50230785111043]
        last_row = [0.4469524747786, 0.61099887445787, 0.53252885275699, 0.24643686103052, 0.42421429774825, 0.49769214888957]
        expected = np.array([first_row, last_row])

        self.assertTrue(np.all(np.isclose(result, expected)))


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
        expected = np.array([256, -294, 974, 645])
        self.assertTrue(np.all(np.equal(expected, result)))

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


class TestRangeShift(unittest.TestCase):

    def test_range_integers(self):
        measurements = np.array([[1.0, 2.0, -3.0], [4.0, -1.0, 3.0]])
        width = 8
        precision = 7
        num_range_bits = 3

        shift = data_utils.select_range_shift(measurements=measurements,
                                              width=width,
                                              precision=precision,
                                              num_range_bits=num_range_bits,
                                              is_unsigned=False)
        self.assertEqual(shift, 2)

    def test_range_mixed_one(self):
        measurements = np.array([[1.5, 2.0, -3.5], [4.75, -1.0, 3.0]])
        width = 5
        precision = 4
        num_range_bits = 3

        shift = data_utils.select_range_shift(measurements=measurements,
                                              width=width,
                                              precision=precision,
                                              num_range_bits=num_range_bits,
                                              is_unsigned=False)
        self.assertEqual(shift, 3)

    def test_range_mixed_two(self):
        measurements = np.array([[1.5, 2.0, -3.5], [4.75, -1.0, 3.0]])
        width = 6
        precision = 4
        num_range_bits = 4

        shift = data_utils.select_range_shift(measurements=measurements,
                                              width=width,
                                              precision=precision,
                                              num_range_bits=num_range_bits,
                                              is_unsigned=False)
        self.assertEqual(shift, 2)

    def test_range_border(self):
        measurements = np.array([[1.869, 1.0]])
        width = 4
        precision = 2
        num_range_bits = 2

        shift = data_utils.select_range_shift(measurements=measurements,
                                              width=width,
                                              precision=precision,
                                              num_range_bits=num_range_bits,
                                              is_unsigned=False)
        self.assertEqual(shift, 0)


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


class TestCalculateBytes(unittest.TestCase):

    def test_byte_block(self):
        # 42 bits -> 6 bytes of data, 1 meta-data, 2 for sequence mask, 16 for IV = 25 bytes -> 32 bytes
        data_bytes = data_utils.calculate_bytes(width=7,
                                                num_features=3,
                                                num_collected=2,
                                                encryption_mode=EncryptionMode.BLOCK,
                                                seq_length=10)
        self.assertEqual(data_bytes, 32)

    def test_byte_stream(self):
        # 42 bits -> 6 bytes of data, 1 meta-data, 2 for sequence mask, 12 for nonce = 21 bytes
        data_bytes = data_utils.calculate_bytes(width=7,
                                                num_features=3,
                                                num_collected=2,
                                                seq_length=9,
                                                encryption_mode=EncryptionMode.STREAM)

        self.assertEqual(data_bytes, 20)

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

        self.assertEqual(message_bytes, len(encrypted))

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

        self.assertEqual(message_bytes, len(encrypted))

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

        self.assertEqual(message_bytes, len(encrypted))

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

        self.assertEqual(message_bytes, len(encrypted))

    def test_group_block(self):
        # 11 bytes of data, 3 meta-data, 2 for sequence mask = 16 bytes -> 16 bytes + 16 byte IV = 32 bytes
        data_bytes = data_utils.calculate_grouped_bytes(widths=[6, 7],
                                                        num_features=3,
                                                        num_collected=4,
                                                        seq_length=10,
                                                        group_size=6,
                                                        encryption_mode=EncryptionMode.BLOCK)
        self.assertEqual(data_bytes, 32)

    def test_group_stream(self):
        # 11 bytes of data, 3 meta-data, 2 for sequence mask, 12 for nonce = 28 bytes
        data_bytes = data_utils.calculate_grouped_bytes(widths=[6, 7],
                                                        num_features=3,
                                                        num_collected=4,
                                                        seq_length=9,
                                                        group_size=6,
                                                        encryption_mode=EncryptionMode.STREAM)

        self.assertEqual(data_bytes, 28)

    def test_group_stream_unbalanced(self):
        # 11 bytes of data, 3 meta-data, 2 for sequence mask, 12 for nonce = 29 bytes
        data_bytes = data_utils.calculate_grouped_bytes(widths=[6, 7],
                                                        num_features=3,
                                                        num_collected=4,
                                                        seq_length=9,
                                                        group_size=6,
                                                        encryption_mode=EncryptionMode.STREAM)

        self.assertEqual(data_bytes, 28)

    def test_group_stream_large(self):
        data_bytes = data_utils.calculate_grouped_bytes(widths=[7, 9],
                                                        num_features=6,
                                                        num_collected=26,
                                                        seq_length=50,
                                                        group_size=132,
                                                        encryption_mode=EncryptionMode.STREAM)
        self.assertEqual(data_bytes, 165)

    def test_group_stream_end_to_end_one(self):
        # Encode and encrypt measurements
        measurements = np.ones(shape=(4, 3))
        collected_indices = [0, 2, 6, 9]
        seq_length = 12
        non_fractional = 2
        widths = [6, 7]
        group_size = 6

        encoded = encode_grouped_measurements(measurements=measurements,
                                              collected_indices=collected_indices,
                                              widths=widths,
                                              seq_length=seq_length,
                                              non_fractional=non_fractional,
                                              group_size=group_size)                                              

        key = get_random_bytes(32)
        encrypted = encrypt(message=encoded, key=key, mode=EncryptionMode.STREAM)

        message_bytes = data_utils.calculate_grouped_bytes(widths=widths,
                                                           num_features=measurements.shape[1],
                                                           num_collected=measurements.shape[0],
                                                           seq_length=seq_length,
                                                           group_size=group_size,
                                                           encryption_mode=EncryptionMode.STREAM)

        self.assertEqual(message_bytes, len(encrypted))

    def test_group_stream_end_to_end_two(self):
        # Encode and encrypt measurements
        measurements = np.ones(shape=(5, 3))
        collected_indices = [0, 2, 6, 9]
        seq_length = 12
        non_fractional = 2
        widths = [6, 7, 7]
        group_size = 6

        encoded = encode_grouped_measurements(measurements=measurements,
                                              collected_indices=collected_indices,
                                              widths=widths,
                                              seq_length=seq_length,
                                              non_fractional=non_fractional,
                                              group_size=group_size)

        key = get_random_bytes(32)
        encrypted = encrypt(message=encoded, key=key, mode=EncryptionMode.STREAM)

        message_bytes = data_utils.calculate_grouped_bytes(widths=widths,
                                                           num_features=measurements.shape[1],
                                                           num_collected=measurements.shape[0],
                                                           seq_length=seq_length,
                                                           group_size=group_size,
                                                           encryption_mode=EncryptionMode.STREAM)

        self.assertEqual(message_bytes, len(encrypted))

    def test_group_block_end_to_end_one(self):
        # Encode and encrypt measurements
        measurements = np.ones(shape=(4, 3))
        collected_indices = [0, 2, 6, 9]
        seq_length = 12
        non_fractional = 2
        widths = [6, 7]
        group_size = 6

        encoded = encode_grouped_measurements(measurements=measurements,
                                              collected_indices=collected_indices,
                                              widths=widths,
                                              seq_length=seq_length,
                                              non_fractional=non_fractional,
                                              group_size=group_size)

        key = get_random_bytes(AES_BLOCK_SIZE)
        encrypted = encrypt(message=encoded, key=key, mode=EncryptionMode.BLOCK)

        message_bytes = data_utils.calculate_grouped_bytes(widths=widths,
                                                           num_features=measurements.shape[1],
                                                           num_collected=measurements.shape[0],
                                                           seq_length=seq_length,
                                                           group_size=group_size,
                                                           encryption_mode=EncryptionMode.BLOCK)

        self.assertEqual(message_bytes, len(encrypted))

    def test_group_block_end_to_end_two(self):
        # Encode and encrypt measurements
        measurements = np.ones(shape=(5, 3))
        collected_indices = [0, 2, 6, 9]
        seq_length = 12
        non_fractional = 2
        widths = [6, 7, 7]
        group_size = 6

        encoded = encode_grouped_measurements(measurements=measurements,
                                              collected_indices=collected_indices,
                                              widths=widths,
                                              seq_length=seq_length,
                                              non_fractional=non_fractional,
                                              group_size=group_size)

        key = get_random_bytes(AES_BLOCK_SIZE)
        encrypted = encrypt(message=encoded, key=key, mode=EncryptionMode.BLOCK)

        message_bytes = data_utils.calculate_grouped_bytes(widths=widths,
                                                           num_features=measurements.shape[1],
                                                           num_collected=measurements.shape[0],
                                                           seq_length=seq_length,
                                                           group_size=group_size,
                                                           encryption_mode=EncryptionMode.BLOCK)

        self.assertEqual(message_bytes, len(encrypted))


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
        self.assertEqual(widths[1], 11)
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
        self.assertEqual(widths[0], 24)
        self.assertEqual(widths[1], 24)

    def test_widths_stream_above(self):
        widths = data_utils.get_group_widths(group_size=6,
                                             num_collected=6,
                                             num_features=3,
                                             seq_length=10,
                                             target_frac=0.5,
                                             standard_width=8,
                                             encryption_mode=EncryptionMode.STREAM)

        self.assertEqual(len(widths), 3)
        self.assertEqual(widths[0], 4)
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


class TestMaxGroups(unittest.TestCase):

    def test_size_one_feature(self):
        target_frac = 0.2
        seq_length = 235
        num_features = 1
        min_width = 3
        group_size = 100
        standard_width = 8
        encryption_mode = EncryptionMode.STREAM

        target_collected = int(seq_length * target_frac)
        target_size = data_utils.calculate_bytes(width=standard_width,
                                                 num_collected=target_collected,
                                                 num_features=num_features,
                                                 seq_length=seq_length,
                                                 encryption_mode=encryption_mode)

        max_collected = data_utils.get_max_collected(seq_length=seq_length,
                                                     num_features=num_features,
                                                     group_size=group_size,
                                                     min_width=min_width,
                                                     target_size=target_size,
                                                     encryption_mode=encryption_mode)

        # Verify the count
        self.assertEqual(max_collected, 113)

        # Verify the number of bytes
        grouped_size = data_utils.calculate_grouped_bytes(widths=[min_width, min_width],
                                                          num_collected=max_collected,
                                                          num_features=num_features,
                                                          group_size=group_size,
                                                          encryption_mode=encryption_mode,
                                                          seq_length=seq_length)

        self.assertLessEqual(grouped_size, target_size)

    def test_size_multi_feature(self):
        target_frac = 0.2
        seq_length = 50
        num_features = 8
        min_width = 3
        group_size = 128
        encryption_mode = EncryptionMode.STREAM

        target_collected = int(seq_length * target_frac)
        target_size = data_utils.calculate_bytes(width=8,
                                                 num_collected=target_collected,
                                                 num_features=num_features,
                                                 seq_length=seq_length,
                                                 encryption_mode=encryption_mode)

        max_collected = data_utils.get_max_collected(seq_length=seq_length,
                                                     num_features=num_features,
                                                     group_size=group_size,
                                                     min_width=min_width,
                                                     target_size=target_size,
                                                     encryption_mode=encryption_mode)

        # Verify the count
        self.assertEqual(max_collected, 25)

        # Verify the number of bytes
        grouped_size = data_utils.calculate_grouped_bytes(widths=[min_width, min_width],
                                                          num_collected=max_collected,
                                                          num_features=num_features,
                                                          group_size=group_size,
                                                          encryption_mode=encryption_mode,
                                                          seq_length=seq_length)

        self.assertLessEqual(grouped_size, target_size)

    def test_size_long_seq(self):
        target_frac = 0.3
        seq_length = 206
        num_features = 3
        min_width = 4
        group_size = 119
        encryption_mode = EncryptionMode.STREAM

        target_collected = int(seq_length * target_frac)
        target_size = data_utils.calculate_bytes(width=8,
                                                 num_collected=target_collected,
                                                 num_features=num_features,
                                                 seq_length=seq_length,
                                                 encryption_mode=encryption_mode)

        max_collected = data_utils.get_max_collected(seq_length=seq_length,
                                                     num_features=num_features,
                                                     group_size=group_size,
                                                     min_width=min_width,
                                                     target_size=target_size,
                                                     encryption_mode=encryption_mode)

        # Verify the count
        self.assertEqual(max_collected, 118)

        # Verify the number of bytes
        grouped_size = data_utils.calculate_grouped_bytes(widths=[min_width, min_width, min_width],
                                                          num_collected=max_collected,
                                                          num_features=num_features,
                                                          group_size=group_size,
                                                          encryption_mode=encryption_mode,
                                                          seq_length=seq_length)

        self.assertLessEqual(grouped_size, target_size)


class TestPruning(unittest.TestCase):

    def test_prune_two(self):
        measurements = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.5, 4.0], [3.5, 3.0]])
        max_collected = 3
        collected_indices = [1, 3, 5, 9, 10]
        seq_length = 12

        # Iter 0 -> Expected Diffs: [0, 2, 2.5, 2.0], Expected Diff Idx: [2, 4, 1, 2]  -> Prune 1
        # iter 1 -> Expected Diffs: [2, 2.5, 2.0], Expected Diff Idx: [4, 1, 2] -> Prune 3

        pruned_features, pruned_indices = data_utils.prune_sequence(measurements=measurements,
                                                                    max_collected=max_collected,
                                                                    collected_indices=collected_indices,
                                                                    seq_length=seq_length)

        expected_features = np.array([[1.0, 1.0], [2.0, 2.0], [3.5, 3.0]])
        expected_indices = [1, 5, 10]

        self.assertTrue(np.all(np.isclose(pruned_features, expected_features)))
        self.assertEqual(pruned_indices, expected_indices)

    def test_prune_final(self):
        measurements = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.5, 4.0], [3.5, 3.0]])
        max_collected = 3
        collected_indices = [1, 3, 5, 7, 10]
        seq_length = 11

        # Iter 0 -> Expected Diffs: [0, 2, 2.5, 2.0], Expected Diff Idx: [2, 2, 3, 1]  -> Prune 1
        # iter 1 -> Expected Diffs: [2, 2.5, 2.0], Expected Diff Idx: [4, 3, 1] -> Prune 4

        pruned_features, pruned_indices = data_utils.prune_sequence(measurements=measurements,
                                                                    max_collected=max_collected,
                                                                    collected_indices=collected_indices,
                                                                    seq_length=seq_length)

        expected_features = np.array([[1.0, 1.0], [2.0, 2.0], [2.5, 4.0]])
        expected_indices = [1, 5, 7]

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

        expected_features = np.array([[1.0, 1.0], [3.0, 3.0]])
        expected_indices = [1, 10]

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
