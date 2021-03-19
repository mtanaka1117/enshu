import numpy as np
import math
from typing import Tuple, List, Dict, Any

from compression import AdaptiveWidth, make_compression
from utils.file_utils import read_pickle_gz
from utils.data_utils import array_to_fp, array_to_float, round_to_block, truncate_to_block, calculate_bytes
from utils.constants import AES_BLOCK_SIZE
from transition_model import TransitionModel


class Policy:

    def __init__(self,
                 transition_model: TransitionModel,
                 target: float,
                 precision: int,
                 width: int,
                 num_features: int,
                 seq_length: int):
        self._transition_model = transition_model
        self._estimate = np.zeros((num_features, 1))  # [D, 1]
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

    @property
    def width_policy(self) -> AdaptiveWidth:
        return self._width_policy

    def reset(self):
        self._estimate = np.zeros((self._num_features, 1))  # [D, 1]

    def transition(self):
        self._estimate = self._transition_model.predict(self._estimate)

    def get_estimate(self) -> np.ndarray:
        return self._estimate

    def quantize_seq(self, measurements: np.ndarray, num_transmitted: int, width: int, should_pad: bool) -> Tuple[np.ndarray, int]:
        non_fractional = self._width - self._precision
        precision = width - non_fractional

        quantized = array_to_fp(arr=measurements,
                                precision=precision,
                                width=width)

        result = array_to_float(fp_arr=quantized,
                                precision=precision)

        total_bytes = calculate_bytes(width=width,
                                      num_transmitted=num_transmitted,
                                      num_features=len(measurements[0]),
                                      should_pad=should_pad)

        return result, total_bytes

    def __str__(self) -> str:
        return 'Policy'

    def transmit(self, measurement: np.ndarray, seq_idx: int) -> int:
        raise NotImplementedError()


class AdaptivePolicy(Policy):

    def __init__(self,
                 transition_model: TransitionModel,
                 threshold: float,
                 target: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int,
                 use_confidence: bool,
                 compression_name: str,
                 compression_params: Dict[str, Any]):
        super().__init__(transition_model=transition_model,
                         precision=precision,
                         width=width,
                         target=target,
                         num_features=num_features,
                         seq_length=seq_length)
        self._threshold = threshold
        self._use_confidence = use_confidence

        self._width_policy = make_compression(name=compression_name,
                                              num_features=num_features,
                                              seq_length=seq_length,
                                              width=width,
                                              target_frac=target,
                                              **compression_params)

        self._max_skip = int(1.0 / target) + 1
        self._current_skip = 0
        self._sample_skip = 0
        self._confidence = 0

    def transmit(self, measurement: np.ndarray, seq_idx: int) -> int:
        if self._use_confidence:
            # Heuristics to guard against over-confident behavior
            if (seq_idx == 0) or (self._sample_skip >= self._max_skip):
                self._estimate = measurement
                self._sample_skip = 0
                self._confidence = 0
                return 1

            # Perform sampling based on confidence
            self._confidence += self._transition_model.confidence(x=measurement)
            
            if self._confidence > self._threshold:
                self._estimate = measurement
                self._sample_skip = 0
                self._confidence = 0
                return 1

            self._sample_skip += 1
            return 0
        else:
            if self._sample_skip > 0:
                self._sample_skip -= 1
                return 0

            diff = np.linalg.norm(self._estimate - measurement, ord=2)
            self._estimate = measurement

            if diff > self._threshold:
                self._current_skip = 0
            else:
                self._current_skip = min(self._current_skip + 1, self._max_skip)

            self._sample_skip = self._current_skip

            return 1

    def reset(self):
        super().reset()
        self._current_skip = 0
        self._sample_skip = 0
        self._confidence = 0

    def quantize_seq(self, measurements: np.ndarray, num_transmitted: int, width: int, should_pad: bool) -> Tuple[np.ndarray, int]:
        # Find the number of non-fractional bits. This part
        # stays constant
        non_fractional = self._width - self._precision

        # Calculate the adaptive fixed-point parameters
        seq_length = len(measurements)
        num_features = len(measurements[0])

        #adaptive_width = max(self._width_policy.get_width(num_transmitted=num_transmitted), non_fractional + 1)
        adaptive_width = max(width, non_fractional + 1)
        adaptive_precision = min(adaptive_width - non_fractional, 20)

        quantized = array_to_fp(measurements,
                                width=adaptive_width,
                                precision=adaptive_precision)
    
        result = array_to_float(quantized, precision=adaptive_precision)

        total_bytes = calculate_bytes(width=adaptive_width,
                                      num_features=self._num_features,
                                      num_transmitted=num_transmitted,
                                      should_pad=should_pad)

        return result, total_bytes

    def __str__(self) -> str:
        return 'Adaptive, {0}'.format(self._width_policy)


class RandomPolicy(Policy):

    def transmit(self, measurement: np.ndarray, seq_idx: int) -> int:
        r = self._rand.uniform()

        if r < self._target or seq_idx == 0:
            self._estimate = measurement
            return 1

        return 0

    def __str__(self) -> str:
        return 'Random'


class UniformPolicy(Policy):
    
    def __init__(self,
                 transition_model: TransitionModel,
                 target: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int):
        super().__init__(transition_model=transition_model,
                         precision=precision,
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

    def transmit(self, measurement: np.ndarray, seq_idx: int) -> int:
        if (seq_idx == 0) or (self._skip_idx < len(self._skip_indices) and seq_idx == self._skip_indices[self._skip_idx]):
            self._estimate = measurement
            self._skip_idx += 1
            return 1

        return 0

    def __str__(self) -> str:
        return 'Uniform'

    def reset(self):
        self._skip_idx = 0


class AllPolicy(Policy):

    def __init__(self,
                 transition_model: TransitionModel,
                 target: float,
                 precision: int,
                 width: int,
                 num_features: int,
                 seq_length: int):
        super().__init__(transition_model=transition_model,
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

    def transmit(self, measurement: np.ndarray, seq_idx: int) -> int:
        self._estimate = measurement
        return 1

    def __str__(self) -> str:
        return 'All'


def make_policy(name: str, transition_model: TransitionModel, seq_length: int, num_features: int, **kwargs: Dict[str, Any]) -> Policy:
    name = name.lower()

    if name == 'random':
        return RandomPolicy(transition_model=transition_model,
                            target=kwargs['target'],
                            precision=kwargs['precision'],
                            width=kwargs['width'],
                            num_features=num_features,
                            seq_length=seq_length)
    elif name == 'adaptive':
        return AdaptivePolicy(transition_model=transition_model,
                              target=kwargs['target'],
                              threshold=kwargs['threshold'],
                              precision=kwargs['precision'],
                              width=kwargs['width'],
                              seq_length=seq_length,
                              num_features=num_features,
                              use_confidence=kwargs['use_confidence'],
                              compression_name=kwargs['compression_name'],
                              compression_params=kwargs['compression_params'])
    elif name == 'uniform':
        return UniformPolicy(transition_model=transition_model,
                             target=kwargs['target'],
                             precision=kwargs['precision'],
                             width=kwargs['width'],
                             num_features=num_features,
                             seq_length=seq_length)
    elif name == 'all':
        return AllPolicy(transition_model=transition_model,
                         target=kwargs['target'],
                         precision=kwargs['precision'],
                         width=kwargs['width'],
                         num_features=num_features,
                         seq_length=seq_length)
    else:
        raise ValueError('Unknown policy with name: {0}'.format(name))
