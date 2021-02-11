import numpy as np
from typing import Tuple, List

from utils.file_utils import read_pickle_gz
from utils.data_utils import array_to_fp, array_to_float


class Policy:

    def __init__(self, transition_path: str, target: float, precision: int, width: int):
        self._transition_mat = read_pickle_gz(transition_path)  # [D, D]
        self._state_size = self._transition_mat.shape[0]
        self._estimate = np.zeros((self._state_size, 1))  # [D, 1]
        self._precision = precision
        self._width = width
        self._target = target
        self._rand = np.random.RandomState(seed=78362)

    def reset(self):
        self._estimate = np.zeros((self._state_size, 1))  # [D, 1]

    def transition(self):
        self._estimate = np.matmul(self._transition_mat, self._estimate)

    def get_estimate(self) -> np.ndarray:
        return self._estimate

    def quantize_seq(self, measurements: List[np.ndarray], num_transmitted: int) -> Tuple[np.ndarray, int]:
        result: List[np.ndaray] = []

        for measurement in measurements:
            quantized = array_to_fp(arr=measurement,
                                    precision=self._precision,
                                    width=self._width)

            unquantized = array_to_float(fp_arr=quantized,
                                         precision=self._precision)

            result.append(np.expand_dims(unquantized, axis=0))

        num_features = measurements[0].shape[0]
        total_bits = num_transmitted * num_features * self._width
        total_bytes = total_bits / 8

        return np.vstack(result), total_bytes

    def transmit(self, measurement: np.ndarray) -> int:
        raise NotImplementedError()


class AdaptivePolicy(Policy):

    def __init__(self, transition_path: str, threshold: float, target: float, precision: int, width: int):
        super().__init__(transition_path=transition_path,
                         precision=precision,
                         width=width,
                         target=target)
        self._threshold = threshold

    def transmit(self, measurement: np.ndarray) -> int:
        diff = np.linalg.norm(self._estimate - measurement, ord=2)

        if diff > self._threshold:
            self._estimate = measurement
            return 1

        return 0

    def quantize_seq(self, measurements: List[np.ndarray], num_transmitted: int) -> Tuple[np.ndarray, int]:
        result: List[np.ndaray] = []

        # Find the number of non-fractional bits. This part
        # stays constant
        non_fractional = self._width - self._precision

        # Calculate the adaptive fixed-point parameters
        seq_length = len(measurements)
        num_features = len(measurements[0])

        target_bits = 8 * self._target * seq_length * num_features  # Target number of bits per sequences (known by design)

        num_transmitted_features = num_transmitted * num_features
        adaptive_width = int(round(target_bits / num_transmitted_features))  # Bits per feature such that we reach the target
        adaptive_precision = int(adaptive_width - non_fractional)

        for measurement in measurements:
            quantized = array_to_fp(arr=measurement,
                                    precision=adaptive_precision,
                                    width=adaptive_width)

            unquantized = array_to_float(fp_arr=quantized,
                                         precision=adaptive_precision)

            result.append(np.expand_dims(unquantized, axis=0))

        total_bits = adaptive_width * num_features * num_transmitted
        total_bytes = total_bits / 8 + 1  # Account for the extra byte required to send the precision

        # TODO: Round to nearest block of 16 bytes to account for encryption

        return np.vstack(result), total_bytes


class RandomPolicy(Policy):

    def transmit(self, measurement: np.ndarray) -> int:
        r = self._rand.uniform()

        if r < self._target:
            self._estimate = measurement
            return 1

        return 0
