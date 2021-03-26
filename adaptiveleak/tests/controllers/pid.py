import unittest
import numpy as np

from adaptiveleak.controllers import PIDController


class PIDControllerTests(unittest.TestCase):

    def test_one_step(self):
        target = 1.0
        estimate = 0.0

        controller = PIDController(kp=0.2, ki=0.1, kd=0.1)

        signal = controller.step(estimate=estimate, target=target)

        # Proportional is 1.0, Integral is 0.5, Derivative is 1.0
        # We start the 'previous' value at zero error
        expected = 0.35
        self.assertTrue(np.isclose(signal, expected))

    def test_two_step(self):
        target = 1.0

        controller = PIDController(kp=0.2, ki=0.1, kd=0.1)
        signal = controller.step(estimate=0.0, target=target)
        signal = controller.step(estimate=signal, target=target)


        # Proportional is (1.0 - 0.35) = 0.65
        # Integral is 0.5 + (1.0 + 0.65) / 2 = 1.325
        # Derivative is (0.65 - 1.0) = -0.35
        # The total is 0.2 * 0.65 + 0.1 * 1.325 + 0.1 * -0.35
        expected = 0.2275
        self.assertTrue(np.isclose(signal, expected))


if __name__ == '__main__':
    unittest.main()
