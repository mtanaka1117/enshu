import numpy as np
import math
from typing import Tuple, List, Dict, Any

from compression import AdaptiveWidth, make_compression
from utils.file_utils import read_pickle_gz
from utils.data_utils import array_to_fp, array_to_float, round_to_block, truncate_to_block, calculate_bytes
from utils.constants import AES_BLOCK_SIZE


class Policy:

    def __init__(self,
                 transition_path: str,
                 target: float,
                 precision: int,
                 width: int,
                 num_features: int,
                 seq_length: int):
        self._transition_mat = read_pickle_gz(transition_path)  # [D, D]
        self._state_size = self._transition_mat.shape[0]
        self._estimate = np.zeros((self._state_size, 1))  # [D, 1]
        self._precision = precision
        self._width = width
        self._num_features = num_features
        self._seq_length = seq_length
        self._target = target

        self._width_policy = make_compression(name='fixed',
                                              num_features=num_features,
                                              seq_length=seq_length,
                                              target_frac=target,
                                              width=width)
        self._rand = np.random.RandomState(seed=78362)

    @property
    def width(self) -> int:
        return self._width

    @property
    def target(self) -> float:
        return self._target

    def reset(self):
        self._estimate = np.zeros((self._state_size, 1))  # [D, 1]

    def transition(self):
        self._estimate = np.matmul(self._transition_mat, self._estimate)

    def get_estimate(self) -> np.ndarray:
        return self._estimate

    def quantize_seq(self, measurements: np.ndarray, num_transmitted: int) -> Tuple[np.ndarray, int]:

        non_fractional = self._width - self._precision
        width = self._width_policy.get_width(num_transmitted=num_transmitted)
        precision = width - non_fractional

        quantized = array_to_fp(arr=measurements,
                                precision=precision,
                                width=width)

        result = array_to_float(fp_arr=quantized,
                                precision=precision)

        total_bytes = calculate_bytes(width=width,
                                      num_transmitted=num_transmitted,
                                      num_features=len(measurements[0]))

        return result, total_bytes

    def __str__(self) -> str:
        return 'Policy'

    def transmit(self, measurement: np.ndarray) -> int:
        raise NotImplementedError()


class AdaptivePolicy(Policy):

    def __init__(self,
                 transition_path: str,
                 threshold: float,
                 target: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int,
                 compression_name: str,
                 compression_params: Dict[str, Any]):
        super().__init__(transition_path=transition_path,
                         precision=precision,
                         width=width,
                         target=target,
                         num_features=num_features,
                         seq_length=seq_length)
        self._threshold = threshold

        self._width_policy = make_compression(name=compression_name,
                                              num_features=num_features,
                                              seq_length=seq_length,
                                              width=width,
                                              target_frac=target,
                                              **compression_params)
        
        #self._width_policy = StochasticBlockWidth(target_frac=target,
        #                                          num_features=num_features,
        #                                          seq_length=seq_length)
        #self._width_policy = PIDWidth(target_frac=target,
        #                                  num_features=num_features,
        #                                  seq_length=seq_length,
        #                                  kp=(1.0 / 32.0),
        #                                  ki=(1.0 / 128.0),
        #                                  kd=(1.0 / 128.0))

    def transmit(self, measurement: np.ndarray) -> int:
        diff = np.linalg.norm(self._estimate - measurement, ord=2)

        if diff > self._threshold:
            self._estimate = measurement
            return 1

        return 0

    def quantize_seq(self, measurements: np.ndarray, num_transmitted: int) -> Tuple[np.ndarray, int]:
        # Find the number of non-fractional bits. This part
        # stays constant
        non_fractional = self._width - self._precision

        # Calculate the adaptive fixed-point parameters
        seq_length = len(measurements)
        num_features = len(measurements[0])

        adaptive_width = self._width_policy.get_width(num_transmitted=num_transmitted)
        adaptive_precision = adaptive_width - non_fractional

        quantized = array_to_fp(measurements,
                                width=adaptive_width,
                                precision=adaptive_precision)
    
        result = array_to_float(quantized, precision=adaptive_precision)

        total_bytes = calculate_bytes(width=adaptive_width,
                                      num_features=self._num_features,
                                      num_transmitted=num_transmitted)

        return result, total_bytes

    def __str__(self) -> str:
        return 'Adaptive, {0}'.format(self._width_policy)


class RandomPolicy(Policy):

    def transmit(self, measurement: np.ndarray) -> int:
        r = self._rand.uniform()

        if r < self._target:
            self._estimate = measurement
            return 1

        return 0

    def __str__(self) -> str:
        return 'Random'


class AllPolicy(Policy):

    def __init__(self,
                 transition_path: str,
                 target: float,
                 precision: int,
                 width: int,
                 num_features: int,
                 seq_length: int):
        super().__init__(transition_path=transition_path,
                         target=target,
                         precision=precision,
                         width=width,
                         num_features=num_features,
                         seq_length=seq_length)

        self._width_policy = make_compression(name='stable',
                                              num_features=num_features,
                                              seq_length=seq_length,
                                              target_frac=target,
                                              width=width)

    def transmit(self, measurement: np.ndarray) -> int:
        self._estimate = measurement
        return 1

#    def quantize_seq(self, measurements: np.ndarray, num_transmitted: int) -> Tuple[np.ndarray, int]:
#        result: List[np.ndaray] = []
#
#        # Find the number of non-fractional bits. This part
#        # stays constant
#        non_fractional = self._width - self._precision
#
#        # Calculate the adaptive fixed-point parameters
#        seq_length, num_features = measurements.shape
#
#        target_bits = 8 * self._target  # Target number of bits per sequences (known by design)
#
#        adaptive_width = int(round(target_bits))
#        adaptive_precision = int(adaptive_width - non_fractional)
#
#        quantized = array_to_fp(arr=measurements,
#                                precision=adaptive_precision,
#                                width=adaptive_width)
#
#        result = array_to_float(quantized, precision=adaptive_precision)
#
#        total_bytes = calculate_bytes(width=adaptive_width,
#                                      num_features=num_features,
#                                      num_transmitted=num_transmitted)
#
#        return result, total_bytes

    def __str__(self) -> str:
        return 'All'


def make_policy(name: str, transition_path: str, seq_length: int, num_features: int, **kwargs: Dict[str, Any]) -> Policy:
    name = name.lower()

    if name == 'random':
        return RandomPolicy(transition_path=transition_path,
                            target=kwargs['target'],
                            precision=kwargs['precision'],
                            width=kwargs['width'],
                            num_features=num_features,
                            seq_length=seq_length)
    elif name == 'adaptive':
        return AdaptivePolicy(transition_path=transition_path,
                              target=kwargs['target'],
                              threshold=kwargs['threshold'],
                              precision=kwargs['precision'],
                              width=kwargs['width'],
                              seq_length=seq_length,
                              num_features=num_features,
                              compression_name=kwargs['compression_name'],
                              compression_params=kwargs['compression_params'])
    elif name == 'all':
        return AllPolicy(transition_path=transition_path,
                         target=kwargs['target'],
                         precision=kwargs['precision'],
                         width=kwargs['width'],
                         num_features=num_features,
                         seq_length=seq_length)
    else:
        raise ValueError('Unknown policy with name: {0}'.format(name))
