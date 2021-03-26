import unittest
import numpy as np
from adaptiveleak.encoding import DeltaEncode


class DeltaEncodingTests(unittest.TestCase):

    def test_gaps(self):
        values = np.array([2.5, 3.7, 2.2, -1.0, 4.0])

        expected_start = 2.5
        expected_gaps = np.array([1.2, -1.5, -3.2, 5.0])

        encoded = DeltaEncode(raw=values)

        self.assertTrue(np.isclose(expected_start, encoded.get_start()))
        self.assertTrue(np.all(np.isclose(expected_gaps, encoded.get_diffs())))
        self.assertTrue(np.all(np.isclose(values, encoded.decode())))

    def test_positive(self):
        rand = np.random.RandomState(seed=57)
        num = 25
        original = rand.uniform(low=0.0, high=5.0, size=num)

        encoder = DeltaEncode(raw=original)

        self.assertTrue(np.all(np.isclose(encoder.decode(), original)))

    def test_negative(self):
        rand = np.random.RandomState(seed=43)
        num = 20
        original = rand.uniform(low=-10.0, high=0.0, size=num)

        encoder = DeltaEncode(raw=original)

        self.assertTrue(np.all(np.isclose(encoder.decode(), original)))

    def test_mixed(self):
        rand = np.random.RandomState(seed=40)
        num = 100
        original = rand.uniform(low=-10.0, high=10.0, size=num)

        encoder = DeltaEncode(raw=original)

        self.assertTrue(np.all(np.isclose(encoder.decode(), original)))


if __name__ == '__main__':
    unittest.main()
