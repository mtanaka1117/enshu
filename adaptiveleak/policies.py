import numpy as np
import math
from typing import Tuple, List, Dict, Any

from controllers import PIDController

from adaptiveleak.utils.data_utils import array_to_fp, array_to_float, round_to_block, truncate_to_block, calculate_bytes, get_group_widths
from adaptiveleak.utils.message import encode_byte_measurements, decode_byte_measurements, encode_grouped_measurements, decode_grouped_measurements
from adaptiveleak.utils.encryption import AES_BLOCK_SIZE, EncryptionType


class Policy:

    def __init__(self,
                 target: float,
                 precision: int,
                 width: int,
                 num_features: int,
                 seq_length: int):
        self._estimate = np.zeros((num_features, 1))  # [D, 1]
        self._precision = precision
        self._width = width
        self._num_features = num_features
        self._seq_length = seq_length
        self._target = target

        self._rand = np.random.RandomState(seed=78362)

        # Track the average number of measurements sent
        self._measurement_count = 0
        self._seq_count = 0

    @property
    def width(self) -> int:
        return self._width

    @property
    def precision(self) -> int:
        return self._precision

    @property
    def seq_length(self) -> int:
        return self._seq_length

    @property
    def num_features(self) -> int:
        return self._num_features

    @property
    def target(self) -> float:
        return self._target

    def collect(self, measurement: np.ndarray):
        self._estimate = np.copy(measurement.reshape(-1, 1))
    
    def reset(self):
        self._estimate = np.zeros((self._num_features, 1))  # [D, 1]

    def encode(self, measurements: np.ndarray, collected_indices: List[int]) -> bytes:
        return encode_byte_measurements(measurements=measurements,
                                        collected_indices=collected_indices,
                                        seq_length=self.seq_length,
                                        precision=self.precision)

    def decode(self, message: bytes) -> Tuple[np.ndarray, List[int]]:
        return decode_byte_measurements(byte_str=message,
                                        seq_length=self.seq_length,
                                        num_features=self.num_features)

    def step(self, count: int, seq_idx: int):
        self._measurement_count += count
        self._seq_count += 1

    def __str__(self) -> str:
        return 'Policy'

    def as_dict(self) -> Dict[str, Any]:
        return {
            'name': str(self),
            'target': self.target,
            'width': self.width,
            'precision': self.precision
        }

    def should_collect(self, measurement: np.ndarray, seq_idx: int) -> bool:
        raise NotImplementedError()


class AdaptivePolicy(Policy):

    def __init__(self,
                 target: float,
                 threshold: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int,
                 encoding_name: str):
        super().__init__(precision=precision,
                         width=width,
                         target=target,
                         num_features=num_features,
                         seq_length=seq_length)

        # Name of the encoding algorithm
        self._encoding_name = encoding_name

        # Variables used to track the adaptive sampling policy
        self._max_skip = int(1.0 / target) + 1
        self._current_skip = 0.0
        self._sample_skip = 0.0

        # Controller automatically set the threshold
        self._pid = PIDController(kp=(1.0 / 16.0),
                                  ki=(1.0 / 32.0),
                                  kd=(1.0 / 128.0))
        self._threshold = threshold

    def should_collect(self, measurement: np.ndarray, seq_idx: int) -> bool:
        if self._sample_skip > 0:
            self._sample_skip -= 1
            return False

        diff = np.linalg.norm(self._estimate - measurement, ord=2)
        self._estimate = measurement

        if diff > self._threshold:
            self._current_skip = 0
        else:
            self._current_skip = min(self._current_skip + 1, self._max_skip)

        self._sample_skip = self._current_skip

        return True

    def reset(self):
        super().reset()
        self._current_skip = 0
        self._sample_skip = 0
        self._confidence = 0

    def step(self, count: int, seq_idx: int):
        super().step(count=count, seq_idx=seq_idx)

        avg = self._measurement_count / (self._seq_count * self._seq_length)
        signal = self._pid.step(estimate=avg, target=self._target)

        if (seq_idx % 20) == 0:
            self._threshold = -1 * signal

    def encode(self, measurements: np.ndarray, collected_indices: List[int]) -> bytes:
        if self._encoding_type == EncodingType.ALL:
            return super().encode(measurements, collected_indices)
        elif self._encoding_type == EncodingType.GROUP:
            num_collected = len(measurements)
            should_pad = self.encryption_type == EncryptionType.BLOCK

            # Get the group widths
            num_groups = get_num_groups(num_transmitted=num_collected,
                                        group_size=self.group_size)

            widths = get_group_widths(num_groups=num_groups,
                                      num_collected=num_collected,
                                      num_features=self.num_features,
                                      seq_length=self.seq_length,
                                      target_frac=self.target,
                                      encryption_type=self.encryption_type)

            return encode_group_measurements(measurements=measurements,
                                             collected_indices=collected_indices,
                                             seq_length=self.seq_length,
                                             widths=widths,
                                             non_fractional=2)
        else:
            raise ValueError('Unknown encoding type {0}'.format(self._encoding_type.name))



#    def quantize_seq(self, measurements: np.ndarray, num_transmitted: int, width: int, should_pad: bool) -> Tuple[np.ndarray, int]:
#        # Find the number of non-fractional bits. This part
#        # stays constant
#        non_fractional = self._width - self._precision
#
#        # Calculate the adaptive fixed-point parameters
#        seq_length = len(measurements)
#        num_features = len(measurements[0])
#
#        adaptive_width = max(width, non_fractional + 1)
#        adaptive_precision = min(adaptive_width - non_fractional, 20)
#
#        quantized = array_to_fp(measurements,
#                                width=adaptive_width,
#                                precision=adaptive_precision)
#
#        result = array_to_float(quantized, precision=adaptive_precision)
#
#        total_bytes = calculate_bytes(width=adaptive_width,
#                                      num_features=self._num_features,
#                                      num_transmitted=num_transmitted,
#                                      should_pad=should_pad)
#
#        return result, total_bytes

    def __str__(self) -> str:
        return 'Adaptive {0}'.format(self._encoding_name)


class RandomPolicy(Policy):

    def should_collect(self, seq_idx: int) -> bool:
        r = self._rand.uniform()

        if r < self._target or seq_idx == 0:
            return True

        return False

    def __str__(self) -> str:
        return 'Random'


class UniformPolicy(Policy):
    
    def __init__(self,
                 target: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int):
        super().__init__(precision=precision,
                         width=width,
                         target=target,
                         num_features=num_features,
                         seq_length=seq_length)
        target_samples = int(math.ceil(target * seq_length))

        skip = max(1.0 / target, 1)
        frac_part = skip - math.floor(skip)

        self._skip_indices: List[int] = []

        index = 0
        while index < seq_length:
            self._skip_indices.append(index)

            if (target_samples - len(self._skip_indices)) == (seq_length - index - 1):
                index += 1
            else:
                r = self._rand.uniform()
                if r > frac_part:
                    index += int(math.floor(skip))
                else:
                    index += int(math.ceil(skip))

        self._skip_indices = self._skip_indices[:target_samples]
        self._skip_idx = 0

    def should_collect(self, seq_idx: int) -> bool:
        if (seq_idx == 0) or (self._skip_idx < len(self._skip_indices) and seq_idx == self._skip_indices[self._skip_idx]):
            self._skip_idx += 1
            return True

        return False

    def __str__(self) -> str:
        return 'Uniform'

    def reset(self):
        self._skip_idx = 0


def run_policy(policy: Policy, sequence: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Executes the policy on the given sequence.

    Args:
        policy: The sampling policy
        sequence: A [T, D] array of features (D) for each element (T)
    Returns:
        A tuple of two elements:
            (1) A [K, D] array of the collected measurements
            (2) The K indices of the collected elements
    """
    assert len(sequence.shape) == 2, 'Must provide a 2d sequence'

    collected_list: List[np.ndarray] = []
    collected_indices: List[int] = []

    for seq_idx in range(sequence.shape[0]):
        
        should_collect = policy.should_collect(seq_idx=seq_idx)

        if should_collect:
            measurement = sequence[seq_idx]
            policy.collect(measurement=measurement)

            collected_list.append(measurement.reshape(1, -1))
            collected_indices.append(seq_idx)

    return np.vstack(collected_list), collected_indices


def make_policy(name: str, seq_length: int, num_features: int, **kwargs: Dict[str, Any]) -> Policy:
    name = name.lower()

    if name == 'random':
        return RandomPolicy(target=kwargs['target'],
                            precision=kwargs['precision'],
                            width=kwargs['width'],
                            num_features=num_features,
                            seq_length=seq_length)
    elif name == 'adaptive':
        return AdaptivePolicy(target=kwargs['target'],
                              threshold=kwargs.get('threshold', 0.0),
                              precision=kwargs['precision'],
                              width=kwargs['width'],
                              seq_length=seq_length,
                              num_features=num_features,
                              use_confidence=kwargs['use_confidence'],
                              compression_name=kwargs['compression_name'],
                              compression_params=kwargs['compression_params'])
    elif name == 'uniform':
        return UniformPolicy(target=kwargs['target'],
                             precision=kwargs['precision'],
                             width=kwargs['width'],
                             num_features=num_features,
                             seq_length=seq_length)
    elif name == 'all':
        return AllPolicy(target=kwargs['target'],
                         precision=kwargs['precision'],
                         width=kwargs['width'],
                         num_features=num_features,
                         seq_length=seq_length)
    else:
        raise ValueError('Unknown policy with name: {0}'.format(name))
