import numpy as np
from collections import deque

from typing import Optional


class PIDController:
    """
    Implementation of a PID controller for feedback-based systems.
    """
    def __init__(self, kp: float, ki: float, kd: float):
        """
        Args:
            kp: The proportional error constant
            ki: The integral error constant
            kd: The derivative error constant
        """
        self._kp = kp
        self._ki = ki
        self._kd = kd

        self._prev_error = 0.0
        self._integral = 0.0

    def step(self, estimate: float, target: float) -> float:
        """
        Updates the PID controller by comparing the estimate to the target value.

        Args:
            estimate: The estimated control value
            target: The target control value
        Returns:
            The control signal to apply to the current value.
        """
        # Compute the error. A positive error means the estimate is too low. 
        # A negative error means the estimate is too high.
        error = target - estimate

        # Compute the proportional term
        prop = error

        # Compute the integral term using the trapezoid rule
        integral = (error + self._prev_error) / 2
        self._integral += integral

        # Compute the derivative term
        derivative = error - self._prev_error

        # print('Error: {0}, Prev Error: {1}, P: {2}, I: {3}, D: {4}'.format(error, self._prev_error, prop, self._integral, derivative))

        # Reset the previous error term
        self._prev_error = error

        return self._kp * prop + self._ki * self._integral + self._kd * derivative
