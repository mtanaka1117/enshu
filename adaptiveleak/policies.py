import numpy as np
import math
import os.path
import time
from collections import deque
from enum import Enum, auto
from typing import Tuple, List, Dict, Any

from adaptiveleak.controllers import PIDController
from adaptiveleak.utils.constants import BITS_PER_BYTE, MIN_WIDTH, SMALL_NUMBER, MAX_WIDTH, SHIFT_BITS, MAX_SHIFT_GROUPS
from adaptiveleak.utils.constants import MIN_SHIFT_GROUPS
from adaptiveleak.utils.data_utils import get_group_widths, get_num_groups, calculate_bytes, pad_to_length, sigmoid
from adaptiveleak.utils.data_utils import prune_sequence, get_max_collected, calculate_grouped_bytes, linear_extrapolate
from adaptiveleak.utils.data_utils import balance_group_size, set_widths, select_range_shifts_array, num_bits_for_value
from adaptiveleak.utils.shifting import merge_shift_groups
from adaptiveleak.utils.message import encode_standard_measurements, decode_standard_measurements
from adaptiveleak.utils.message import encode_grouped_measurements, decode_grouped_measurements
from adaptiveleak.utils.message import encode_stable_measurements, decode_stable_measurements
from adaptiveleak.utils.encryption import AES_BLOCK_SIZE, EncryptionMode, CHACHA_NONCE_LEN
from adaptiveleak.utils.file_utils import read_json, read_pickle_gz


MARGIN = 0.0
MAX_ITER = 100


class EncodingMode(Enum):
    STANDARD = auto()
    GROUP = auto()


class Policy:

    def __init__(self,
                 target: float,
                 precision: int,
                 width: int,
                 num_features: int,
                 seq_length: int,
                 encryption_mode: EncryptionMode,
                 should_compress: bool):
        self._estimate = np.zeros((num_features, 1))  # [D, 1]
        self._precision = precision
        self._width = width
        self._num_features = num_features
        self._seq_length = seq_length
        self._target = target
        self._encryption_mode = encryption_mode
        self._should_compress = should_compress

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
    def non_fractional(self) -> int:
        return self.width - self.precision

    @property
    def seq_length(self) -> int:
        return self._seq_length

    @property
    def num_features(self) -> int:
        return self._num_features

    @property
    def target(self) -> float:
        return self._target

    @property
    def encryption_mode(self) -> EncryptionMode:
        return self._encryption_mode

    @property
    def should_compress(self) -> bool:
        return self._should_compress
    
    def collect(self, measurement: np.ndarray):
        self._estimate = np.copy(measurement.reshape(-1, 1))
    
    def reset(self):
        self._estimate = np.zeros((self._num_features, 1))  # [D, 1]

    def encode(self, measurements: np.ndarray, collected_indices: List[int]) -> bytes:
        return encode_standard_measurements(measurements=measurements,
                                            collected_indices=collected_indices,
                                            seq_length=self.seq_length,
                                            precision=self.precision,
                                            width=self.width,
                                            should_compress=self.should_compress)

    def decode(self, message: bytes) -> Tuple[np.ndarray, List[int]]:
        return decode_standard_measurements(byte_str=message,
                                            seq_length=self.seq_length,
                                            num_features=self.num_features,
                                            precision=self.precision,
                                            width=self.width,
                                            should_compress=self.should_compress)

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
            'precision': self.precision,
            'encryption_mode': self._encryption_mode.name,
            'should_compress': self.should_compress
        }

    def should_collect(self, seq_idx: int) -> bool:
        raise NotImplementedError()


class AdaptivePolicy(Policy):

    def __init__(self,
                 target: float,
                 threshold: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int,
                 max_skip: int,
                 encryption_mode: EncryptionMode,
                 encoding_mode: EncodingMode,
                 should_compress: bool):
        super().__init__(precision=precision,
                         width=width,
                         target=target,
                         num_features=num_features,
                         seq_length=seq_length,
                         encryption_mode=encryption_mode,
                         should_compress=should_compress)

        # Name of the encoding algorithm
        self._encoding_mode = encoding_mode

        # Variables used to track the adaptive sampling policy
        self._max_skip = int(1.0 / target) + max_skip
        self._current_skip = 0
        self._sample_skip = 0

        self._threshold = threshold

    @property
    def encoding_mode(self) -> EncodingMode:
        return self._encoding_mode

    def reset(self):
        super().reset()
        self._current_skip = 0
        self._sample_skip = 0

    def encode(self, measurements: np.ndarray, collected_indices: List[int]) -> bytes:
        if self.encoding_mode == EncodingMode.STANDARD:
            return super().encode(measurements, collected_indices)
        elif self.encoding_mode == EncodingMode.GROUP:
            # Get the maximum number of collected measurements to still
            # meet the target size
            target_bytes = calculate_bytes(width=self.width,
                                           num_collected=int(self.target * self.seq_length),
                                           num_features=self.num_features,
                                           seq_length=self.seq_length,
                                           encryption_mode=self.encryption_mode)

            # Conservatively Estimate the meta-data bytes associated with stable encoding
            size_width = num_bits_for_value(len(collected_indices))
            size_bytes = int(math.ceil((size_width * MAX_SHIFT_GROUPS) / BITS_PER_BYTE))
            mask_bytes = int(math.ceil(self.seq_length / BITS_PER_BYTE))

            shift_bytes = 1 + MAX_SHIFT_GROUPS + size_bytes
            metadata_bytes = shift_bytes + mask_bytes

            if self.encryption_mode == EncryptionMode.STREAM:
                metadata_bytes += CHACHA_NONCE_LEN
            else:
                metadata_bytes += AES_BLOCK_SIZE

            # Compute the target number of data bytes
            target_data_bytes = target_bytes - metadata_bytes
            target_data_bits = (target_data_bytes - MAX_SHIFT_GROUPS) * self.width

            # Estimate the maximum number of measurements we can collect
            max_features = int(target_data_bits / MIN_WIDTH)
            max_collected = int(max_features / self.num_features)

            # Prune measurements if needed
            measurements, collected_indices = prune_sequence(measurements=measurements,
                                                             collected_indices=collected_indices,
                                                             max_collected=max_collected,
                                                             seq_length=self.seq_length)

            flattened = measurements.T.reshape(-1)
            min_width = int(target_data_bits / (self.num_features * len(collected_indices)))

            # Select the range shifts
            shifts = select_range_shifts_array(measurements=flattened,
                                               width=min_width,
                                               precision=min_width - self.non_fractional,
                                               num_range_bits=SHIFT_BITS)

            # Merge the shift groups
            merged_shifts, group_sizes = merge_shift_groups(values=flattened,
                                                            shifts=shifts,
                                                            max_num_groups=MAX_SHIFT_GROUPS)

            # Re-calculate the meta-data size based on the given shift groups. Smaller
            # ranges allow for greater savings.
            size_width = num_bits_for_value(max(group_sizes))
            size_bytes = int(math.ceil((size_width * MAX_SHIFT_GROUPS) / BITS_PER_BYTE))

            shift_bytes = 1 + MAX_SHIFT_GROUPS + size_bytes
            metadata_bytes = shift_bytes + mask_bytes

            if self.encryption_mode == EncryptionMode.STREAM:
                metadata_bytes += CHACHA_NONCE_LEN
            else:
                metadata_bytes += AES_BLOCK_SIZE

            target_data_bytes = target_bytes - metadata_bytes

            # Set the group sizes
            group_widths = set_widths(group_sizes, target_bytes=target_data_bytes, start_width=MIN_WIDTH)

            encoded = encode_stable_measurements(measurements=measurements,
                                                 collected_indices=collected_indices,
                                                 widths=group_widths,
                                                 shifts=merged_shifts,
                                                 group_sizes=group_sizes,
                                                 non_fractional=self.non_fractional,
                                                 seq_length=self.seq_length)

            diff = (target_bytes - CHACHA_NONCE_LEN) - len(encoded)

            if self.encryption_mode == EncryptionMode.STREAM:
                return pad_to_length(encoded, length=target_bytes - CHACHA_NONCE_LEN)

            return encoded
        else:
            raise ValueError('Unknown encoding type {0}'.format(self.encoding_mode.name))

    def decode(self, message: bytes) -> Tuple[np.ndarray, List[int]]:
        if self.encoding_mode == EncodingMode.STANDARD:
            return super().decode(message)
        elif self.encoding_mode == EncodingMode.GROUP:
            non_fractional = self.width - self.precision
            max_group_size = max(int(BITS_PER_BYTE * AES_BLOCK_SIZE), 1)

            return decode_stable_measurements(encoded=message,
                                              seq_length=self.seq_length,
                                              num_features=self.num_features,
                                              non_fractional=non_fractional)
        else:
            raise ValueError('Unknown encoding type {0}'.format(self.encoding_mode.name))

    def __str__(self) -> str:
        return 'adaptive_{0}'.format(self._encoding_mode.name.lower())

    def as_dict(self) -> Dict[str, Any]:
        policy_dict = super().as_dict()
        policy_dict['encoding'] = self._encoding_mode.name
        return policy_dict


class AdaptiveHeuristic(AdaptivePolicy):

    def should_collect(self, seq_idx: int) -> bool:
        if self._sample_skip > 0:
            self._sample_skip -= 1
            return False

        return True

    def collect(self, measurement: np.ndarray):
        diff = np.sum(np.abs(self._estimate - measurement))
        self._estimate = measurement

        if diff >= self._threshold:
            self._current_skip = 0
        else:
            self._current_skip = min(self._current_skip + 1, self._max_skip)

        self._sample_skip = self._current_skip

    def __str__(self) -> str:
        return 'adaptive_heuristic_{0}'.format(self._encoding_mode.name.lower())


class AdaptiveLiteSense(AdaptivePolicy):

    def __init__(self,
                 target: float,
                 threshold: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int,
                 max_skip: int,
                 encryption_mode: EncryptionMode,
                 encoding_mode: EncodingMode,
                 should_compress: bool):
        super().__init__(target=target,
                         threshold=threshold,
                         precision=precision,
                         width=width,
                         seq_length=seq_length,
                         num_features=num_features,
                         max_skip=max_skip,
                         encryption_mode=encryption_mode,
                         encoding_mode=encoding_mode,
                         should_compress=should_compress)
        self._alpha = 0.7
        self._beta = 0.7

        self._mean = np.zeros(shape=(num_features, 1))  # [D, 1]
        self._dev = np.zeros(shape=(num_features, 1))

    def should_collect(self, seq_idx: int) -> bool:
        if (seq_idx == 0) or (self._sample_skip >= self._current_skip):
            return True

        self._sample_skip += 1
        return False

    def collect(self, measurement: np.ndarray):
        updated_mean = (1.0 - self._alpha) * self._mean + self._alpha * measurement
        updated_dev = (1.0 - self._beta) * self._dev + self._beta * np.abs(updated_mean - measurement)

        diff = np.sum(updated_dev - self._dev)

        if diff >= self._threshold:
            self._current_skip = max(self._current_skip - 1, 0)
        else:
            self._current_skip = min(self._current_skip + 1, self._max_skip)

        self._estimate = measurement

        self._mean = updated_mean
        self._dev = updated_dev

        self._sample_skip = 0

    def reset(self):
        super().reset()
        self._mean = np.zeros(shape=(self.num_features, 1))  # [D, 1]
        self._dev = np.zeros(shape=(self.num_features, 1))

    def __str__(self) -> str:
        return 'adaptive_litesense_{0}'.format(self._encoding_mode.name.lower())


class AdaptiveDeviation(AdaptiveLiteSense):
    
    def collect(self, measurement: np.ndarray):
        self._mean = (1.0 - self._alpha) * self._mean + self._alpha * measurement
        self._dev = (1.0 - self._beta) * self._dev + self._beta * np.abs(self._mean - measurement)

        norm = np.sum(self._dev)

        if norm > self._threshold:
            self._current_skip = max(int(self._current_skip / 2), 0) 
        else:
            self._current_skip = min(self._current_skip + 1, self._max_skip)

        self._estimate = measurement
        self._sample_skip = 0

    def __str__(self) -> str:
        return 'adaptive_deviation_{0}'.format(self._encoding_mode.name.lower())


class SkipRNN(AdaptivePolicy):

    def __init__(self,
                 target: float,
                 threshold: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int,
                 encryption_mode: EncryptionMode,
                 encoding_mode: EncodingMode,
                 should_compress: bool,
                 dataset_name: str):
        super().__init__(target=target,
                         threshold=threshold,
                         precision=precision,
                         width=width,
                         seq_length=seq_length,
                         num_features=num_features,
                         max_skip=0,
                         encryption_mode=encryption_mode,
                         encoding_mode=encoding_mode,
                         should_compress=should_compress)

        # Fetch the parameters
        model_file = os.path.join('saved_models', dataset_name, 'skip_rnn', 'skip-rnn-{0}.pkl.gz'.format(int(target * 100)))
        model_weights = read_pickle_gz(model_file)['trainable_vars']

        # Unpack the model parameters
        self._W_gates = model_weights['rnn-cell/W-gates:0'].T
        self._b_gates = model_weights['rnn-cell/b-gates:0'].T

        #self._W_candidate = model_weights['rnn-cell/W-candidate:0'].T
        #self._b_candidate = model_weights['rnn-cell/b-candidate:0'].T

        self._W_state = model_weights['rnn-cell/W-state:0'].T
        self._b_state = model_weights['rnn-cell/b-state:0'].T

        #self._alpha = sigmoid(model_weights['rnn-cell/alpha:0'])
        #self._beta = sigmoid(model_weights['rnn-cell/beta:0'])

        # Unpack the normalization object
        scaler = read_pickle_gz(model_file)['metadata']['scaler']
        self._mean = np.expand_dims(scaler.mean_, axis=-1) # [K, 1]
        self._scale = np.expand_dims(scaler.scale_, axis=-1)  # [K, 1]

        # Initialize the state
        self._state_size = self._W_state.shape[1]

        self._initial_state = model_weights['initial-hidden-state:0'].T
       
        self._state = self._initial_state
        self._cum_update_prob = 1.0  # Cumulative update prob
        self._update_prob = 0.0  # Update prob from the previous step (avoid re-computation)

    def should_collect(self, seq_idx: int) -> bool:
        if (self._cum_update_prob >= self._threshold):
            self._cum_update_prob = 0.0            
            return True
        else:
            self._cum_update_prob = min(self._cum_update_prob + self._update_prob, 1.0)
            return False

    def collect(self, measurement: np.ndarray):
        assert len(measurement.shape) in (1, 2), 'Must prove a 1d or 2d measurement'

        if len(measurement.shape) == 1:
            measurement = np.expand_dims(measurement, axis=-1)

        # Normalize the measurements
        measurement = (measurement - self._mean) / self._scale

        # Compute the Fast RNN Update
        stacked = np.concatenate([measurement, self._state], axis=0)  # [K + D, 1]
        #candidate = np.tanh(np.matmul(self._W_gates, stacked) + self._b_gates)  # [D, 1]

        #self._state = self._alpha * candidate + self._beta * self._state

        gates = np.matmul(self._W_gates, stacked) + self._b_gates

        update_gate, candidate = gates[:self._state_size], gates[self._state_size:]

        update_gate = sigmoid(update_gate + 1)
        candidate = np.tanh(candidate)
        #reset_gate = sigmoid(reset_gate)

        # Compute the candidate state
        #stacked = np.concatenate([measurement, reset_gate * self._state], axis=0)
        #candidate = np.tanh(self._W_candidate.dot(stacked) + self._b_candidate)

        # Compute the next state
        self._state = (1.0 - update_gate) * candidate + update_gate * self._state

        # Compute the update probabilities
        self._update_prob = sigmoid(self._W_state.dot(self._state) + self._b_state)
        self._cum_update_prob = self._update_prob

    def reset(self):
        self._state = self._initial_state
        self._cum_update_prob = 1.0  # Cumulative update prob
        self._update_prob = 0.0  # Update prob from the previous step (avoid re-computation)

    def __str__(self) -> str:
        return 'skip_rnn_{0}'.format(self._encoding_mode.name.lower())


class AdaptiveJitter(AdaptivePolicy):

    def __init__(self,
                 target: float,
                 threshold: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int,
                 encryption_mode: EncryptionMode,
                 encoding_mode: EncodingMode,
                 should_compress: bool):
        super().__init__(target=target,
                         threshold=threshold,
                         precision=precision,
                         width=width,
                         seq_length=seq_length,
                         num_features=num_features,
                         encryption_mode=encryption_mode,
                         encoding_mode=encoding_mode,
                         should_compress=should_compress)

        # Queue of captured measurements over time
        self._captured: deque = deque()

        # Queue of the indices of the captured measurements
        self._captured_idx: deque = deque()

        self._window = 3
        self._seq_idx = 0 

        self._measurement_sum = np.zeros(shape=(num_features,))
        self._time_sum = 0

    def should_collect(self, seq_idx: int) -> bool:
        self._seq_idx = seq_idx

        if (seq_idx > 0) and (self._sample_skip < self._current_skip) and (seq_idx < (self._seq_length - 1)):
            self._sample_skip += 1
            return False

        return True

    def collect(self, measurement: np.ndarray):
        # Add measurement and index to the queue
        self._captured.append(measurement)
        self._captured_idx.append(self._seq_idx)

        # Update the sum values
        self._measurement_sum += measurement
        self._time_sum += self._seq_idx

        while len(self._captured) > self._window:
            removed_measurement = self._captured.popleft()
            self._measurement_sum -= removed_measurement

            removed_time = self._captured_idx.popleft()
            self._time_sum -= removed_time

        # Update the mean values
        mean_idx = self._time_sum / len(self._captured_idx)
        mean_element = self._measurement_sum / len(self._captured)

        # Calculate the differences
        idx_diff = sum(np.square(idx - mean_idx) for idx in self._captured_idx)
        measurement_diff = np.zeros_like(measurement, dtype=float)

        for element, idx in zip(self._captured, self._captured_idx):
            measurement_diff += (idx - mean_idx) * (element - mean_element)

        # Form the best-fit parameters
        slope = measurement_diff / (idx_diff + SMALL_NUMBER)
        intercept = mean_element - slope * mean_idx

        # Calculate the jitter metric
        jitter = 0.0
        median_idx = (max(self._captured_idx) + min(self._captured_idx)) / 2
        median_element = intercept + slope * median_idx

        for element in self._captured:
            jitter += np.sum(np.abs(median_element - element))

        # Update the skipping window
        if jitter > self._threshold:
            self._current_skip = max(int(self._current_skip / 2), 0)
        else:
            self._current_skip = min(self._current_skip + 1, self._max_skip)

        self._sample_skip = 0

    def reset(self):
        super().reset()
        self._captured = deque()
        self._captured_idx = deque()
        self._seq_idx = 0
        self._time_sum = 0
        self._measurement_sum = np.zeros(shape=(self.num_features,))

    def __str__(self) -> str:
        return 'adaptive_jitter_{0}'.format(self._encoding_mode.name.lower())


class AdaptiveSlope(AdaptiveJitter):

    def __init__(self,
                 target: float,
                 threshold: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int,
                 encryption_mode: EncryptionMode,
                 encoding_mode: EncodingMode,
                 should_compress: bool):
        super().__init__(target=target,
                         threshold=threshold,
                         precision=precision,
                         width=width,
                         seq_length=seq_length,
                         num_features=num_features,
                         encryption_mode=encryption_mode,
                         encoding_mode=encoding_mode,
                         should_compress=should_compress)
        self._window = 2
        self._current_skip = int(self._max_skip / 2)
        self._prev_slope = np.zeros((num_features, ))

    def collect(self, measurement: np.ndarray):
        # Add measurement and index to the queue
        self._captured.append(measurement)
        self._captured_idx.append(self._seq_idx)

        # Update the sum values
        while len(self._captured) > self._window:
            self._captured.popleft()
            self._captured_idx.popleft()

        self._sample_skip = 0

        if len(self._captured) < self._window:
            self._current_skip = int(self._max_skip / 2)
            return

        # Compute the slope based on the previous two values
        feature_diff = self._captured[-1] - self._captured[-2]
        time_diff = self._captured_idx[-1] - self._captured_idx[-2]
        slope = feature_diff / time_diff

        error = np.sum(np.abs(slope - self._prev_slope))
        self._prev_slope = slope

        # Update the skipping window
        if error > self._threshold:
            self._current_skip = max(int(self._current_skip / 2), 0)
        else:
            self._current_skip = min(self._current_skip * 2 + 1, self._max_skip)

        self._sample_skip = 0

    def __str__(self):
        return 'adaptive_slope_{0}'.format(self._encoding_mode.name.lower())


class AdaptiveLinear(AdaptiveJitter):

    def __init__(self,
                 target: float,
                 threshold: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int,
                 encryption_mode: EncryptionMode,
                 encoding_mode: EncodingMode,
                 should_compress: bool):
        super().__init__(target=target,
                         threshold=threshold,
                         precision=precision,
                         width=width,
                         seq_length=seq_length,
                         num_features=num_features,
                         encryption_mode=encryption_mode,
                         encoding_mode=encoding_mode,
                         should_compress=should_compress)
        self._window = 2
        self._current_skip = int(self._max_skip / 2)

    def collect(self, measurement: np.ndarray):
        # Add measurement and index to the queue
        self._captured.append(measurement)
        self._captured_idx.append(self._seq_idx)

        self._sample_skip = 0

        if len(self._captured) <= self._window:
            self._current_skip = int(self._max_skip / 2)
            return

        # Compute the slope based on the previous two values
        feature_diff = self._captured[-2] - self._captured[-3]
        time_diff = self._captured_idx[-2] - self._captured_idx[-3]
        slope = feature_diff / time_diff

        step = self._captured_idx[-1] - self._captured_idx[-3]
        projected = slope * step + self._captured[-3]

        error = np.sum(np.abs(projected - measurement))

        if (error < self._threshold):
            #self._current_skip = self._current_skip * 2 + 1
            self._current_skip = self._max_skip
        else:
            #self._current_skip -= 1
            self._current_skip = int(self._current_skip / 2)

        self._current_skip = max(min(self._current_skip, self._max_skip), 0)

        # Update the sum values
        while len(self._captured) > self._window:
            self._captured.popleft()
            self._captured_idx.popleft()

        ## Update the mean values
        #mean_idx = self._time_sum / len(self._captured_idx)
        #mean_element = self._measurement_sum / len(self._captured)

        ## Calculate the differences
        #idx_diff = sum(np.square(idx - mean_idx) for idx in self._captured_idx)
        #measurement_diff = np.zeros_like(measurement)

        #for element, idx in zip(self._captured, self._captured_idx):
        #    measurement_diff += (idx - mean_idx) * (element - mean_element)

        ## Form the best-fit parameters
        #slope = measurement_diff / (idx_diff + SMALL_NUMBER)
        #delta = int(math.floor(self._threshold / (np.sum(np.abs(slope)) + SMALL_NUMBER)))
        #
        #self._current_skip = min(delta, self._max_skip)
        #self._sample_skip = 0

    def __str__(self):
        return 'adaptive_linear'.format(self._encoding_mode.name.lower())


class RandomPolicy(Policy):

    def should_collect(self, seq_idx: int) -> bool:
        r = self._rand.uniform()

        if r < self._target or seq_idx == 0:
            return True

        return False

    def __str__(self) -> str:
        return 'random'


class UniformPolicy(Policy):
    
    def __init__(self,
                 target: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int,
                 encryption_mode: EncryptionMode,
                 should_compress: bool):
        super().__init__(precision=precision,
                         width=width,
                         target=target,
                         num_features=num_features,
                         seq_length=seq_length,
                         encryption_mode=encryption_mode,
                         should_compress=should_compress)
        target_samples = int(target * seq_length)

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
        return 'uniform'

    def reset(self):
        super().reset()
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


def make_policy(name: str,
                seq_length: int,
                num_features: int,
                encryption_mode: EncryptionMode,
                target: float,
                dataset: str,
                should_compress: bool,
                **kwargs: Dict[str, Any]) -> Policy:
    name = name.lower()

    # Look up the data-specific precision and width
    quantize_path = os.path.join('datasets', dataset, 'quantize.json')
    quantize_dict = read_json(quantize_path)
    precision = quantize_dict['precision']
    width = quantize_dict['width']
    max_skip = quantize_dict.get('max_skip', 1)

    if name == 'random':
        return RandomPolicy(target=target + MARGIN,
                            precision=precision,
                            width=width,
                            num_features=num_features,
                            seq_length=seq_length,
                            encryption_mode=encryption_mode,
                            should_compress=should_compress)
    elif name == 'uniform':
        return UniformPolicy(target=target + MARGIN,
                             precision=precision,
                             width=width,
                             num_features=num_features,
                             seq_length=seq_length,
                             encryption_mode=encryption_mode,
                             should_compress=should_compress)
    elif name.startswith('adaptive'):
        # Look up the threshold path
        threshold_path = os.path.join('saved_models', dataset, 'thresholds.pkl.gz')

        if not os.path.exists(threshold_path):
            print('WARNING: No threshold path exists. Defaulting to 0.0')
            threshold = 0.0
        else:
            thresholds = read_pickle_gz(threshold_path)

            if (name not in thresholds) or (target not in thresholds[name]):
                print('WARNING: No threshold path exists. Defaulting to 0.0')
                threshold = 0.0
            else:
                threshold = thresholds[name][target]

        if name == 'adaptive_heuristic':
            cls = AdaptiveHeuristic
        elif name == 'adaptive_litesense':
            cls = AdaptiveLiteSense
        elif name == 'adaptive_deviation':
            cls = AdaptiveDeviation
        elif name == 'adaptive_jitter':
            cls = AdaptiveJitter
        elif name == 'adaptive_linear':
            cls = AdaptiveLinear
        elif name == 'adaptive_slope':
            cls = AdaptiveSlope
        else:
            raise ValueError('Unknown adaptive policy with name: {0}'.format(name))

        return cls(target=target,
                   threshold=threshold,
                   precision=precision,
                   width=width,
                   seq_length=seq_length,
                   num_features=num_features,
                   max_skip=max_skip,
                   encryption_mode=encryption_mode,
                   encoding_mode=EncodingMode[str(kwargs['encoding']).upper()],
                   should_compress=should_compress)
    elif name == 'skip_rnn':
        return SkipRNN(target=target,
                       threshold=0.5,
                       precision=precision,
                       width=width,
                       seq_length=seq_length,
                       num_features=num_features,
                       encryption_mode=encryption_mode,
                       encoding_mode=EncodingMode[str(kwargs['encoding']).upper()],
                       should_compress=should_compress,
                       dataset_name=dataset)
    else:
        raise ValueError('Unknown policy with name: {0}'.format(name))
