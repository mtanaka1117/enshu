import numpy as np
import math
from typing import Tuple, List, Dict, Any

from utils.file_utils import read_pickle_gz
from utils.data_utils import array_to_fp, array_to_float, round_to_block, truncate_to_block


AES_BLOCK_SIZE = 16


class Policy:

    def __init__(self, transition_path: str, target: float, precision: int, width: int):
        self._transition_mat = read_pickle_gz(transition_path)  # [D, D]
        self._state_size = self._transition_mat.shape[0]
        self._estimate = np.zeros((self._state_size, 1))  # [D, 1]
        self._precision = precision
        self._width = width
        self._target = target
        self._rand = np.random.RandomState(seed=78362)

    @property
    def width(self) -> int:
        return self._width

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

        total_bytes = round_to_block(total_bytes, AES_BLOCK_SIZE)

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

        # Set the target quantities. We add one to account for sending
        # the precision & width (both are small and can fit within 4 bits)
        target_bytes = self._target * seq_length * num_features - 1
        target_bits = 8 * self._target * seq_length * num_features  # Target number of bits per sequences (known by design)

        upper_bytes = round_to_block(target_bytes, AES_BLOCK_SIZE)
        lower_bytes = truncate_to_block(target_bytes, AES_BLOCK_SIZE)

        num_transmitted_features = num_transmitted * num_features

        # Get the adaptive width using stochastic rounding
        adaptive_width = int(target_bits / num_transmitted_features)

        to_bytes = lambda w: (1 + ((w * num_transmitted_features) / 8))
        recovered_bytes = to_bytes(adaptive_width)

        r = self._rand.uniform()
        # rate = (recovered_bytes - lower_bytes) / (upper_bytes - lower_bytes)
        
        if r < 0.5:
            # Round up to the upper bytes threshold
            while to_bytes(adaptive_width) <= upper_bytes:
                adaptive_width += 1

            adaptive_width -= 1
        else:
            # Round down to the lower bytes threshold
            while to_bytes(adaptive_width) > lower_bytes:
                adaptive_width -= 1

        adaptive_precision = int(adaptive_width - non_fractional)

        for measurement in measurements:
            quantized = array_to_fp(arr=measurement,
                                    precision=adaptive_precision,
                                    width=adaptive_width)

            unquantized = array_to_float(fp_arr=quantized,
                                         precision=adaptive_precision)

            result.append(np.expand_dims(unquantized, axis=0))

        total_bytes = to_bytes(adaptive_width)

        # print('Total Bytes: {0}, Adaptive Width: {1}, Target Bytes: {2} ({3}, {4})'.format(total_bytes, adaptive_width, target_bytes, lower_bytes, upper_bytes))

        total_bytes = round_to_block(total_bytes, block_size=AES_BLOCK_SIZE)

        return np.vstack(result), total_bytes


class RandomPolicy(Policy):

    def transmit(self, measurement: np.ndarray) -> int:
        r = self._rand.uniform()

        if r < self._target:
            self._estimate = measurement
            return 1

        return 0


class AllPolicy(Policy):

    def transmit(self, measurement: np.ndarray) -> int:
        self._estimate = measurement
        return 1

    def quantize_seq(self, measurements: List[np.ndarray], num_transmitted: int) -> Tuple[np.ndarray, int]:
        result: List[np.ndaray] = []

        # Find the number of non-fractional bits. This part
        # stays constant
        non_fractional = self._width - self._precision

        # Calculate the adaptive fixed-point parameters
        seq_length = len(measurements)
        num_features = len(measurements[0])

        target_bits = 8 * self._target  # Target number of bits per sequences (known by design)

        adaptive_width = int(round(target_bits))
        adaptive_precision = int(adaptive_width - non_fractional)

        for measurement in measurements:
            quantized = array_to_fp(arr=measurement,
                                    precision=adaptive_precision,
                                    width=adaptive_width)

            unquantized = array_to_float(fp_arr=quantized,
                                         precision=adaptive_precision)

            result.append(np.expand_dims(unquantized, axis=0))

        total_bits = adaptive_width * num_features * num_transmitted
        total_bytes = (total_bits / 8)

        total_bytes = round_to_block(total_bytes, block_size=AES_BLOCK_SIZE)

        return np.vstack(result), total_bytes



def make_policy(name: str, transition_path: str, **kwargs: Dict[str, Any]) -> Policy:
    name = name.lower()

    if name == 'random':
        return RandomPolicy(transition_path=transition_path,
                            target=kwargs['target'],
                            precision=kwargs['precision'],
                            width=kwargs['width'])
    elif name == 'adaptive':
        return AdaptivePolicy(transition_path=transition_path,
                              target=kwargs['target'],
                              threshold=kwargs['threshold'],
                              precision=kwargs['precision'],
                              width=kwargs['width'])
    elif name == 'all':
        return AllPolicy(transition_path=transition_path,
                         target=kwargs['target'],
                         precision=kwargs['precision'],
                         width=kwargs['width'])
    else:
        raise ValueError('Unknown policy with name: {0}'.format(name))

