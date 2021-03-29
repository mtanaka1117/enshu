import unittest
import numpy as np
from adaptiveleak.utils import data_utils


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

        self.assertEqual(293, data_utils.to_fixed_point(0.28700833, precision=10, width=12))
        self.assertEqual(974, data_utils.to_fixed_point(0.95151288, precision=10, width=12))
        self.assertEqual(645, data_utils.to_fixed_point(0.63029945, precision=10, width=12))

        self.assertEqual(4, data_utils.to_fixed_point(0.28700833, precision=4, width=6))
        self.assertEqual(60, data_utils.to_fixed_point(0.95151288, precision=6, width=10))
        self.assertEqual(161, data_utils.to_fixed_point(0.63029945, precision=8, width=15))

    def test_to_fp_neg(self):
        self.assertEqual(-64, data_utils.to_fixed_point(-0.25, precision=8, width=10))
        self.assertEqual(-256, data_utils.to_fixed_point(-0.25, precision=10, width=12))

        self.assertEqual(-293, data_utils.to_fixed_point(-0.28700833, precision=10, width=12))
        self.assertEqual(-974, data_utils.to_fixed_point(-0.95151288, precision=10, width=12))
        self.assertEqual(-645, data_utils.to_fixed_point(-0.63029945, precision=10, width=12))

        self.assertEqual(-4, data_utils.to_fixed_point(-0.28700833, precision=4, width=6))
        self.assertEqual(-60, data_utils.to_fixed_point(-0.95151288, precision=6, width=10))
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

    def test_array_to_fp(self):
        array = np.array([0.25, -0.28700833, 0.95151288, 0.63029945])
        result = data_utils.array_to_fp(array, precision=10, width=12)
        expected = np.array([256, -293, 974, 645])
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

    def test_array_to_float(self):
        array = np.array([-256, 293, 974, -645])
        result = data_utils.array_to_float(array, precision=10)
        expected = np.array([-0.25, 0.2861328125, 0.951171875, -0.6298828125])
        self.assertTrue(np.all(np.isclose(expected, result)))


class TestExtrapolation(unittest.TestCase):

    def test_one(self):
        prev = np.array([1, 1, 1, 1], dtype=float)
        curr = np.array([2, 2, 2, 2], dtype=float)

        predicted = data_utils.linear_extrapolate(prev=prev, curr=curr, delta=1)
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

        predicted = data_utils.linear_extrapolate(prev=prev, curr=curr, delta=0.25)
        
        t2 = np.ones_like(m) * 1.5
        expected = m * t2 + b

        self.assertTrue(np.all(np.isclose(predicted, expected)))


if __name__ == '__main__':
    unittest.main()
