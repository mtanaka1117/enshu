import numpy as np

import math
import os.path
import time
from collections import deque, OrderedDict
from enum import Enum, auto
from typing import Tuple, List, Dict, Any, Optional


from adaptiveleak.energy_systems import EnergyUnit, convert_rate_to_energy, get_group_target_bytes
from adaptiveleak.utils.constants import BITS_PER_BYTE, MIN_WIDTH, SMALL_NUMBER, MAX_WIDTH, SHIFT_BITS, MAX_SHIFT_GROUPS
from adaptiveleak.utils.constants import MIN_SHIFT_GROUPS, PERIOD, LENGTH_SIZE, BT_FRAME_SIZE, MAX_SHIFT_GROUPS_FACTOR
from adaptiveleak.utils.data_utils import get_group_widths, get_num_groups, calculate_bytes, pad_to_length, sigmoid, truncate_to_block, round_to_block
from adaptiveleak.utils.data_utils import prune_sequence, calculate_grouped_bytes, set_widths, select_range_shifts_array, num_bits_for_value
from adaptiveleak.utils.shifting import merge_shift_groups
from adaptiveleak.utils.message import encode_standard_measurements, decode_standard_measurements
from adaptiveleak.utils.message import encode_stable_measurements, decode_stable_measurements
from adaptiveleak.utils.encryption import AES_BLOCK_SIZE, CHACHA_NONCE_LEN
from adaptiveleak.utils.file_utils import read_json, read_pickle_gz, read_json_gz
from adaptiveleak.utils.data_types import EncodingMode, EncryptionMode, PolicyType, PolicyResult, CollectMode


class Policy:

    def __init__(self,
                 collection_rate: float,
                 precision: int,
                 width: int,
                 num_features: int,
                 seq_length: int,
                 encryption_mode: EncryptionMode,
                 encoding_mode: EncodingMode,
                 collect_mode: CollectMode,
                 should_compress: bool):
        self._estimate = np.zeros((num_features, ))  # [D]
        self._precision = precision
        self._width = width
        self._num_features = num_features
        self._seq_length = seq_length
        self._collection_rate = collection_rate
        self._encryption_mode = encryption_mode
        self._encoding_mode = encoding_mode
        self._collect_mode = collect_mode
        self._should_compress = should_compress

        self._rand = np.random.RandomState(seed=78362)

        # Track the average number of measurements sent
        self._measurement_count = 0
        self._seq_count = 0

        # Make the energy unit
        self._energy_unit = EnergyUnit(policy_type=self.policy_type,
                                       encoding_mode=self.encoding_mode,
                                       encryption_mode=self.encryption_mode,
                                       collect_mode=self.collect_mode,
                                       seq_length=self.seq_length,
                                       num_features=num_features,
                                       period=PERIOD)

        self._target_bytes = calculate_bytes(width=self.width,
                                             num_collected=int(self.collection_rate * self.seq_length),
                                             num_features=self.num_features,
                                             seq_length=self.seq_length,
                                             encryption_mode=self.encryption_mode)

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
    def collection_rate(self) -> float:
        return self._collection_rate

    @property
    def encryption_mode(self) -> EncryptionMode:
        return self._encryption_mode

    @property
    def encoding_mode(self) -> EncodingMode:
        return self._encoding_mode

    @property
    def collect_mode(self) -> CollectMode:
        return self._collect_mode

    @property
    def energy_unit(self) -> EnergyUnit:
        return self._energy_unit

    @property
    def should_compress(self) -> bool:
        return self._should_compress

    @property
    def target_bytes(self) -> int:
        return self._target_bytes

    def collect(self, measurement: np.ndarray):
        self._estimate = np.copy(measurement.reshape(-1))  # [D]

    def reset(self):
        self._estimate = np.zeros((self._num_features, ))  # [D]

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

    def get_energy(self, num_collected: int, num_bytes: int) -> float:
        return self.energy_unit.get_energy(num_collected=num_collected,
                                           num_bytes=num_bytes)

    def __str__(self) -> str:
        return '{0}-{1}-{2}-{3}'.format(self.policy_type.name.lower(), self.encoding_mode.name.lower(), self.encryption_mode.name.lower(), self.collect_mode.name.lower())

    def as_dict(self) -> Dict[str, Any]:
        return {
            'policy_name': self.policy_type.name,
            'collection_rate': self.collection_rate,
            'width': self.width,
            'precision': self.precision,
            'encryption_mode': self.encryption_mode.name,
            'encoding_mode': self.encoding_mode.name,
            'collect_mode': self.collect_mode.name,
            'should_compress': self.should_compress
        }

    @property
    def policy_type(self) -> PolicyType:
        raise NotImplementedError()

    def should_collect(self, seq_idx: int) -> bool:
        raise NotImplementedError()


class AdaptivePolicy(Policy):

    def __init__(self,
                 collection_rate: float,
                 threshold: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int,
                 use_min_skip: bool,
                 max_skip: int,
                 encryption_mode: EncryptionMode,
                 encoding_mode: EncodingMode,
                 collect_mode: CollectMode,
                 should_compress: bool,
                 max_collected: Optional[int] = None):
        super().__init__(precision=precision,
                         width=width,
                         collection_rate=collection_rate,
                         num_features=num_features,
                         seq_length=seq_length,
                         encryption_mode=encryption_mode,
                         encoding_mode=encoding_mode,
                         collect_mode=collect_mode,
                         should_compress=should_compress)
        # Variables used to track the adaptive sampling policy
        self._max_skip = int(1.0 / collection_rate) + max_skip

        self._min_skip = 0

        if use_min_skip:
            if collection_rate < 0.31:
                self._min_skip = 2
            elif collection_rate < 0.51:
                self._min_skip = 1
            else:
                self._min_skip = 0

        assert self._max_skip > self._min_skip, 'Must have a max skip > min_skip'

        self._current_skip = 0
        self._sample_skip = 0

        self._threshold = threshold

        self._energy_per_seq = convert_rate_to_energy(collection_rate=collection_rate,
                                                      width=self.width,
                                                      encryption_mode=encryption_mode,
                                                      collect_mode=collect_mode,
                                                      seq_length=seq_length,
                                                      num_features=num_features)

        # Set the maximum number of shift groups
        target_features = int(self.collection_rate * seq_length) * num_features
        num_groups = int(round(MAX_SHIFT_GROUPS_FACTOR * target_features))
        self._max_num_groups = max(min(MAX_SHIFT_GROUPS, num_groups), MIN_SHIFT_GROUPS)

        if self.encoding_mode == EncodingMode.PADDED:
            num_collected = max_collected if max_collected is not None else self.seq_length

            self._target_bytes = calculate_bytes(width=self.width,
                                                 num_collected=num_collected,
                                                 num_features=self.num_features,
                                                 encryption_mode=self.encryption_mode,
                                                 seq_length=self.seq_length)
        elif self.encoding_mode != EncodingMode.STANDARD:
            self._target_bytes = get_group_target_bytes(width=self.width,
                                                        collection_rate=self.collection_rate,
                                                        num_features=self.num_features,
                                                        seq_length=self.seq_length,
                                                        encryption_mode=self.encryption_mode,
                                                        energy_unit=self.energy_unit,
                                                        target_energy=self._energy_per_seq)

    @property
    def max_skip(self) -> int:
        return self._max_skip

    @property
    def min_skip(self) -> int:
        return self._min_skip

    @property
    def max_num_groups(self) -> int:
        return self._max_num_groups

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def target_bytes(self) -> int:
        return self._target_bytes

    def set_threshold(self, threshold: float):
        self._threshold = threshold

    def reset(self):
        super().reset()
        self._current_skip = 0
        self._sample_skip = 0

    def encode(self, measurements: np.ndarray, collected_indices: List[int]) -> bytes:
        if self.encoding_mode == EncodingMode.STANDARD:
            return super().encode(measurements, collected_indices)
        elif self.encoding_mode == EncodingMode.PADDED:
            encoded = super().encode(measurements, collected_indices)

            if self.encryption_mode == EncryptionMode.STREAM:
                return pad_to_length(encoded, length=self.target_bytes - CHACHA_NONCE_LEN - LENGTH_SIZE)
            elif self.encryption_mode == EncryptionMode.BLOCK:
                return pad_to_length(encoded, length=self.target_bytes - AES_BLOCK_SIZE - LENGTH_SIZE)
            else:
                raise ValueError('Unknown encryption mode {0}'.format(self.encryption_mode.name))

        elif self.encoding_mode == EncodingMode.PRUNED:
            metadata_bytes = int(math.ceil(self.seq_length / BITS_PER_BYTE)) + LENGTH_SIZE

            if self.encryption_mode == EncryptionMode.STREAM:
                metadata_bytes += CHACHA_NONCE_LEN
            else:
                metadata_bytes += AES_BLOCK_SIZE

            # Compute the target number of data bytes
            target_data_bytes = self.target_bytes - metadata_bytes
            target_data_bits = target_data_bytes * BITS_PER_BYTE

            # Estimate the maximum number of measurements we can collect
            max_features = int(target_data_bits / self.width)
            max_collected = int(max_features / self.num_features)

            # Prune measurements if needed
            measurements, collected_indices = prune_sequence(measurements=measurements,
                                                             collected_indices=collected_indices,
                                                             max_collected=max_collected,
                                                             seq_length=self.seq_length)

            # Encode the pruned sequence
            encoded = super().encode(measurements, collected_indices)

            # Pad the sequence if needed
            if self.encryption_mode == EncryptionMode.STREAM:
                return pad_to_length(encoded, length=self.target_bytes - CHACHA_NONCE_LEN - LENGTH_SIZE)
            elif self.encryption_mode == EncryptionMode.BLOCK:
                return pad_to_length(encoded, length=self.target_bytes - AES_BLOCK_SIZE - LENGTH_SIZE)
            else:
                raise ValueError('Unknown encryption mode {0}'.format(self.encryption_mode.name))

        elif self.encoding_mode in (EncodingMode.GROUP, EncodingMode.GROUP_UNSHIFTED, EncodingMode.SINGLE_GROUP):
            target_bytes = self._target_bytes

            # Conservatively Estimate the meta-data bytes associated with stable encoding
            size_width = num_bits_for_value(len(collected_indices))
            size_bytes = int(math.ceil((size_width * self.max_num_groups) / BITS_PER_BYTE))
            mask_bytes = int(math.ceil(self.seq_length / BITS_PER_BYTE))

            shift_bytes = 1 + self.max_num_groups + size_bytes
            metadata_bytes = shift_bytes + mask_bytes + LENGTH_SIZE

            if self.encryption_mode == EncryptionMode.STREAM:
                metadata_bytes += CHACHA_NONCE_LEN
            else:
                metadata_bytes += AES_BLOCK_SIZE

            # Compute the target number of data bytes
            target_data_bytes = target_bytes - metadata_bytes
            target_data_bits = (target_data_bytes - self.max_num_groups) * BITS_PER_BYTE

            assert target_data_bits > 0, 'Must have a positive number of target data bits'

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
            min_width = min(min_width, max(MAX_WIDTH, self.width))

            group_sizes: List[int] = []
            merged_shifts: List[int] = []

            if self.encoding_mode == EncodingMode.GROUP:
                # Select the range shifts
                shifts = select_range_shifts_array(measurements=flattened,
                                                   old_width=self.width,
                                                   old_precision=self.precision,
                                                   new_width=min_width,
                                                   num_range_bits=SHIFT_BITS)

                # Merge the shift groups
                merged_shifts, group_sizes = merge_shift_groups(values=flattened,
                                                                shifts=shifts,
                                                                max_num_groups=self.max_num_groups)
            elif self.encoding_mode == EncodingMode.GROUP_UNSHIFTED:
                # Set the group sizes 'evenly'
                features_per_group = int(round(len(flattened) / self.max_num_groups))

                feature_count = 0
                for group_idx in range(self.max_num_groups - 1):
                    group_sizes.append(features_per_group)
                    feature_count += features_per_group

                # Include the remaining elements in the last group
                group_sizes.append(len(flattened) - feature_count)

                # For the 'un-shifted' variant, we set all the shift values to zero
                merged_shifts = [0 for _ in group_sizes]
            elif self.encoding_mode == EncodingMode.SINGLE_GROUP:
                group_sizes.append(len(flattened))  # Use a single group with no shift
                merged_shifts.append(0)
            else:
                raise ValueError('Unknown encoding mode: {0}'.format(self.encoding_mode))

            # Re-calculate the meta-data size based on the given shift groups. Smaller
            # ranges allow for greater savings.
            size_width = num_bits_for_value(max(group_sizes))
            size_bytes = int(math.ceil((size_width * self.max_num_groups) / BITS_PER_BYTE))

            shift_bytes = 1 + self.max_num_groups + size_bytes
            metadata_bytes = shift_bytes + mask_bytes + LENGTH_SIZE

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

            if self.encryption_mode == EncryptionMode.STREAM:
                return pad_to_length(encoded, length=target_bytes - CHACHA_NONCE_LEN - LENGTH_SIZE)
            elif self.encryption_mode == EncryptionMode.BLOCK:
                return pad_to_length(encoded, length=target_bytes - AES_BLOCK_SIZE - LENGTH_SIZE)
            else:
                raise ValueError('Unknown encryption mode {0}'.format(self.encryption_mode.name))
        else:
            raise ValueError('Unknown encoding type {0}'.format(self.encoding_mode.name))

    def decode(self, message: bytes) -> Tuple[np.ndarray, List[int]]:
        if self.encoding_mode in (EncodingMode.STANDARD, EncodingMode.PRUNED, EncodingMode.PADDED):
            return super().decode(message)
        elif self.encoding_mode in (EncodingMode.GROUP, EncodingMode.GROUP_UNSHIFTED, EncodingMode.SINGLE_GROUP):
            non_fractional = self.width - self.precision
            max_group_size = max(int(BITS_PER_BYTE * AES_BLOCK_SIZE), 1)

            return decode_stable_measurements(encoded=message,
                                              seq_length=self.seq_length,
                                              num_features=self.num_features,
                                              non_fractional=non_fractional)
        else:
            raise ValueError('Unknown encoding type {0}'.format(self.encoding_mode.name))


class AdaptiveHeuristic(AdaptivePolicy):

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.ADAPTIVE_HEURISTIC

    def should_collect(self, seq_idx: int) -> bool:
        if (self._sample_skip > 0):
            self._sample_skip -= 1
            return False

        return True

    def collect(self, measurement: np.ndarray):
        if len(measurement.shape) >= 2:
            measurement = measurement.reshape(-1)

        diff = np.sum(np.abs(self._estimate - measurement))
        self._estimate = measurement

        if diff >= self.threshold:
            self._current_skip = self.min_skip
        else:
            self._current_skip = min(self._current_skip + 1, self.max_skip)

        self._sample_skip = self._current_skip


class AdaptiveLiteSense(AdaptivePolicy):

    def __init__(self,
                 collection_rate: float,
                 threshold: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int,
                 max_skip: int,
                 use_min_skip: bool,
                 encryption_mode: EncryptionMode,
                 encoding_mode: EncodingMode,
                 collect_mode: CollectMode,
                 should_compress: bool):
        super().__init__(collection_rate=collection_rate,
                         threshold=threshold,
                         precision=precision,
                         width=width,
                         seq_length=seq_length,
                         num_features=num_features,
                         max_skip=max_skip,
                         use_min_skip=use_min_skip,
                         encryption_mode=encryption_mode,
                         encoding_mode=encoding_mode,
                         collect_mode=collect_mode,
                         should_compress=should_compress)
        self._alpha = 0.7
        self._beta = 0.7

        self._mean = np.zeros(shape=(num_features, ))  # [D]
        self._dev = np.zeros(shape=(num_features, ))

    def should_collect(self, seq_idx: int) -> bool:
        if (seq_idx == 0) or (self._sample_skip >= self._current_skip):
            return True

        self._sample_skip += 1
        return False

    def collect(self, measurement: np.ndarray):
        if len(measurement.shape) >= 2:
            measurement = measurement.reshape(-1)  # [D]

        updated_mean = (1.0 - self._alpha) * self._mean + self._alpha * measurement
        updated_dev = (1.0 - self._beta) * self._dev + self._beta * np.abs(updated_mean - measurement)

        diff = np.sum(updated_dev - self._dev)

        if diff >= self.threshold:
            self._current_skip = max(self._current_skip - 1, 0)
        else:
            self._current_skip = min(self._current_skip + 1, self._max_skip)

        self._estimate = measurement

        self._mean = updated_mean
        self._dev = updated_dev

        self._sample_skip = 0

    def reset(self):
        super().reset()
        self._mean = np.zeros(shape=(self.num_features, ))  # [D]
        self._dev = np.zeros(shape=(self.num_features, ))  # [D]


class AdaptiveDeviation(AdaptiveLiteSense):

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.ADAPTIVE_DEVIATION

    def collect(self, measurement: np.ndarray):
        self._mean = (1.0 - self._alpha) * self._mean + self._alpha * measurement
        self._dev = (1.0 - self._beta) * self._dev + self._beta * np.abs(self._mean - measurement)

        norm = np.sum(self._dev)

        if norm > self.threshold:
            self._current_skip = max(int(self._current_skip / 2), self.min_skip)
        else:
            self._current_skip = min(self._current_skip + 1, self.max_skip)

        self._estimate = measurement
        self._sample_skip = 0


class SkipRNN(AdaptivePolicy):

    def __init__(self,
                 collection_rate: float,
                 threshold: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int,
                 encryption_mode: EncryptionMode,
                 encoding_mode: EncodingMode,
                 collect_mode: CollectMode,
                 should_compress: bool,
                 dataset_name: str):
        # Enforce that the threshold is in [0, 1]
        assert threshold >= 0 and threshold <= 1, 'Must have a threshold in [0, 1]'

        super().__init__(collection_rate=collection_rate,
                         threshold=threshold,
                         precision=precision,
                         width=width,
                         seq_length=seq_length,
                         num_features=num_features,
                         max_skip=0,
                         use_min_skip=False,
                         encryption_mode=encryption_mode,
                         encoding_mode=encoding_mode,
                         collect_mode=collect_mode,
                         should_compress=should_compress)

        # Set the target energy level to that of the Standard Skip RNN. We do not
        # enforce the same energy budgets on Skip RNNs due to their high energy cost.
        # In this manner, they cannot be directly compared to other policies.
        target_collected = int(collection_rate * seq_length)

        sent_bytes = calculate_bytes(width=self.width,
                                     num_collected=target_collected,
                                     num_features=num_features,
                                     seq_length=seq_length,
                                     encryption_mode=self.encryption_mode)

        self._energy_per_seq = self.energy_unit.get_energy(num_collected=target_collected,
                                                           num_bytes=sent_bytes,
                                                           use_noise=False)

        # Re-calculate the target bytes based on the updated energy per sequence
        if self.encoding_mode == EncodingMode.PADDED:
            self._target_bytes = calculate_bytes(width=self.width,
                                                 num_collected=self.seq_length,
                                                 num_features=self.num_features,
                                                 encryption_mode=self.encryption_mode,
                                                 seq_length=self.seq_length)
        elif self.encoding_mode != EncodingMode.STANDARD:
            self._target_bytes = get_group_target_bytes(width=self.width,
                                                        collection_rate=self.collection_rate,
                                                        num_features=self.num_features,
                                                        seq_length=self.seq_length,
                                                        encryption_mode=self.encryption_mode,
                                                        energy_unit=self.energy_unit,
                                                        target_energy=self._energy_per_seq)

        # Fetch the parameters
        dir_name = os.path.dirname(__file__)
        model_file = os.path.join(dir_name, 'saved_models', dataset_name, 'skip_rnn', 'skip-rnn-{0}.pkl.gz'.format(int(collection_rate * 100)))
        model_weights = read_pickle_gz(model_file)['trainable_vars']

        # Unpack the model parameters
        self._W_gates = model_weights['rnn-cell/W-gates:0'].T
        self._b_gates = model_weights['rnn-cell/b-gates:0'].T

        self._W_state = model_weights['rnn-cell/W-state:0'].T
        self._b_state = model_weights['rnn-cell/b-state:0'].T

        # Unpack the normalization object
        scaler = read_pickle_gz(model_file)['metadata']['scaler']
        self._mean = np.expand_dims(scaler.mean_, axis=-1)  # [K, 1]
        self._scale = np.expand_dims(scaler.scale_, axis=-1)  # [K, 1]

        # Initialize the state
        self._state_size = self._W_state.shape[1]

        self._initial_state = model_weights['initial-hidden-state:0'].T

        self._state = self._initial_state
        self._cum_update_prob = 1.0  # Cumulative update prob
        self._update_prob = 0.0  # Update prob from the previous step (avoid re-computation)

        self._seq_idx = 0

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.SKIP_RNN

    def set_threshold(self, threshold: float):
        assert threshold >= 0 and threshold <= 1, 'Must have threshold in [0, 1]. Got {0:.5f}'.format(threshold)
        self._threshold = threshold

    def should_collect(self, seq_idx: int) -> bool:
        self._seq_idx += 1

        if (self._cum_update_prob >= self.threshold):
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

        # Compute the UGRNN Update
        stacked = np.concatenate([measurement, self._state], axis=0)  # [K + D, 1]
        gates = np.matmul(self._W_gates, stacked) + self._b_gates

        update_gate, candidate = gates[:self._state_size], gates[self._state_size:]

        update_gate = sigmoid(update_gate + 1)
        candidate = np.tanh(candidate)

        # Compute the next state
        self._state = (1.0 - update_gate) * candidate + update_gate * self._state

        # Compute the update probabilities
        update_prob = self._W_state.dot(self._state) + self._b_state
        self._update_prob = sigmoid(update_prob)
        self._cum_update_prob = self._update_prob

    def reset(self):
        self._state = self._initial_state
        self._cum_update_prob = 1.0  # Cumulative update prob
        self._update_prob = 0.0  # Update prob from the previous step (avoid re-computation)
        self._seq_idx = 0


class RandomPolicy(Policy):

    def __init__(self,
                 collection_rate: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int,
                 encryption_mode: EncryptionMode,
                 collect_mode: CollectMode,
                 should_compress: bool):
        super().__init__(precision=precision,
                         width=width,
                         collection_rate=collection_rate,
                         num_features=num_features,
                         seq_length=seq_length,
                         encryption_mode=encryption_mode,
                         collect_mode=collect_mode,
                         encoding_mode=EncodingMode.STANDARD,
                         should_compress=should_compress)
        self._rand_indices = list(range(1, self.seq_length))
        self._indices = [0]

        self._collect_idx = 0
        self._num_to_collect = int(self.collection_rate * self.seq_length) - 1  # Always collect index 0

        self.reset()

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.RANDOM

    def should_collect(self, seq_idx: int) -> bool:
        if (self._collect_idx < len(self._indices)) and (self._indices[self._collect_idx] == seq_idx):
            self._collect_idx += 1
            return True

        return False

    def reset(self):
        self._indices = [0]
        
        idx_to_collect = np.sort(self._rand.choice(self._rand_indices, size=self._num_to_collect, replace=False)).tolist()
        self._indices.extend(idx_to_collect)

        self._collect_idx = 0


class UniformPolicy(Policy):

    def __init__(self,
                 collection_rate: float,
                 precision: int,
                 width: int,
                 seq_length: int,
                 num_features: int,
                 encryption_mode: EncryptionMode,
                 collect_mode: CollectMode,
                 should_compress: bool):
        super().__init__(precision=precision,
                         width=width,
                         collection_rate=collection_rate,
                         num_features=num_features,
                         seq_length=seq_length,
                         encryption_mode=encryption_mode,
                         encoding_mode=EncodingMode.STANDARD,
                         collect_mode=collect_mode,
                         should_compress=should_compress)
        target_samples = int(collection_rate * seq_length)

        skip = max(1.0 / collection_rate, 1)
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

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.UNIFORM

    def should_collect(self, seq_idx: int) -> bool:
        if (self._skip_idx < len(self._skip_indices) and seq_idx == self._skip_indices[self._skip_idx]):
            self._skip_idx += 1
            return True

        return False

    def reset(self):
        super().reset()
        self._skip_idx = 0


class BudgetWrappedPolicy(Policy):

    def __init__(self,
                 name: str,
                 seq_length: int,
                 num_features: int,
                 encryption_mode: str,
                 collect_mode: str,
                 collection_rate: float,
                 dataset: str,
                 should_compress: bool,
                 **kwargs: Dict[str, Any]):
        # Make the internal policy
        self._policy = make_policy(name=name,
                                   seq_length=seq_length,
                                   num_features=num_features,
                                   encryption_mode=encryption_mode,
                                   collect_mode=collect_mode,
                                   collection_rate=collection_rate,
                                   dataset=dataset,
                                   should_compress=should_compress,
                                   **kwargs)

        # Call the base constructor to set the internal fields
        super().__init__(seq_length=seq_length,
                         num_features=num_features,
                         encryption_mode=self._policy.encryption_mode,
                         encoding_mode=self._policy.encoding_mode,
                         collect_mode=self._policy.collect_mode,
                         width=self._policy.width,
                         precision=self._policy.precision,
                         collection_rate=collection_rate,
                         should_compress=should_compress)

        # Convert the target rate into an energy rate per sequences
        self._energy_per_seq = convert_rate_to_energy(collection_rate=collection_rate,
                                                      width=self.width,
                                                      encryption_mode=self.encryption_mode,
                                                      collect_mode=self.collect_mode,
                                                      seq_length=self.seq_length,
                                                      num_features=self.num_features)

        # Counters for tracking the energy consumption
        self._consumed_energy = 0.0
        self._num_sequences: Optional[int] = None
        self._budget: Optional[float] = None

        # Get the data distributions for possible random sequence generation
        dirname = os.path.dirname(__file__)
        distribution_path = os.path.join(dirname, 'datasets', dataset, 'distribution.json')
        distribution = read_json(distribution_path)

        self._data_mean = np.array(distribution['mean'])
        self._data_std = np.array(distribution['std'])

    @property
    def policy_type(self) -> PolicyType:
        return self._policy.policy_type

    @property
    def energy_per_seq(self) -> float:
        return self._energy_per_seq

    @property
    def budget(self) -> Optional[float]:
        return self._budget

    @property
    def consumed_energy(self) -> float:
        return self._consumed_energy

    def set_threshold(self, threshold: float):
        self._policy.set_threshold(threshold)

    def encode(self, measurements: np.ndarray, collected_indices: List[int]) -> bytes:
        return self._policy.encode(measurements=measurements,
                                   collected_indices=collected_indices)

    def decode(self, message: bytes) -> Tuple[np.ndarray, List[int]]:
        return self._policy.decode(message=message)

    def should_collect(self, seq_idx: int) -> bool:
        return self._policy.should_collect(seq_idx=seq_idx)    

    def collect(self, measurement: np.ndarray):
        self._policy.collect(measurement=measurement)

    def reset(self):
        self._policy.reset()

    def init_for_experiment(self, num_sequences: int):
        self._consumed_energy = 0.0
        self._num_sequences = num_sequences
        self._budget = self.energy_per_seq * num_sequences

    def consume_energy(self, num_collected: int, num_bytes: int) -> float:
        energy = self.energy_unit.get_energy(num_collected=num_collected,
                                             num_bytes=num_bytes,
                                             use_noise=False)
        self._consumed_energy += energy
        return energy

    def has_exhausted_budget(self) -> bool:
        assert self._budget is not None, 'Must call init_for_experiment() first'
        return self._consumed_energy > self._budget

    def get_random_sequence(self) -> np.ndarray:
        rand_list: List[np.ndarray] = []

        for m, s in zip(self._data_mean, self._data_std):
            val = self._rand.normal(loc=m, scale=s, size=self.seq_length)  # [T]
            rand_list.append(np.expand_dims(val, axis=-1))

        return np.concatenate(rand_list, axis=-1)  # [T, D]

    def as_dict(self) -> Dict[str, Any]:
        result = super().as_dict()
        result['budget'] = self._budget
        result['energy_per_seq'] = self.energy_per_seq
        return result


def run_policy(policy: BudgetWrappedPolicy, sequence: np.ndarray, should_enforce_budget: bool) -> PolicyResult:
    """
    Executes the policy on the given sequence.

    Args:
        policy: The sampling policy
        sequence: A [T, D] array of features (D) for each element (T)
        should_enforce_budget: Whether to enforce the current energy budget
    Returns:
        A tuple of three elements:
            (1) A [K, D] array of the collected measurements
            (2) The K indices of the collected elements
            (3) The encoded message as a byte string
            (4) The energy required for this sequence
    """
    assert len(sequence.shape) == 2, 'Must provide a 2d sequence'

    # Reset all internal per-sequence counters
    policy.reset()

    # Unpack the shape
    seq_length, num_features = sequence.shape

    if should_enforce_budget and policy.has_exhausted_budget():
        rand_measurements = policy.get_random_sequence()
        return PolicyResult(measurements=rand_measurements,
                            collected_indices=list(range(seq_length)),
                            num_collected=seq_length,
                            energy=0.0,
                            num_bytes=0,
                            encoded=bytes())

    # Lists to hold the results
    collected_list: List[np.ndarray] = []
    collected_indices: List[int] = []

    # Execute the policy on the given sequence
    for seq_idx in range(seq_length):
        should_collect = policy.should_collect(seq_idx=seq_idx)

        if should_collect:
            measurement = sequence[seq_idx]
            policy.collect(measurement=measurement)

            collected_list.append(measurement.reshape(1, -1))
            collected_indices.append(seq_idx)

    # Stack collected features into a numpy array
    collected = np.vstack(collected_list)  # [K, D]

    # Encode the results into a byte string
    encoded = policy.encode(measurements=collected,
                            collected_indices=collected_indices)

    # Compute the number of bytes accounting for the length and encryption algorithm
    num_bytes = len(encoded)

    if policy.encryption_mode == EncryptionMode.STREAM:
        num_bytes += CHACHA_NONCE_LEN  # Add the Nonce
    elif policy.encryption_mode == EncryptionMode.BLOCK:
        num_bytes = round_to_block(num_bytes, block_size=AES_BLOCK_SIZE)  # Pad for the block cipher
        num_bytes += AES_BLOCK_SIZE  # Add in the IV
    else:
        raise ValueError('Unknown encryption mode: {0}'.format(policy.encoding_mode.name.lower()))

    # Include the length field
    num_bytes += LENGTH_SIZE

    # Compute the energy required
    energy = policy.consume_energy(num_collected=len(collected_indices),
                                   num_bytes=num_bytes)

    if should_enforce_budget and policy.has_exhausted_budget():
        rand_measurements = policy.get_random_sequence()
        policy._consumed_energy = policy._budget + SMALL_NUMBER

        return PolicyResult(measurements=rand_measurements,
                            collected_indices=list(range(seq_length)),
                            num_collected=seq_length,
                            energy=0.0,
                            num_bytes=0,
                            encoded=bytes())

    return PolicyResult(measurements=collected,
                        collected_indices=collected_indices,
                        num_collected=len(collected_indices),
                        encoded=encoded,
                        num_bytes=num_bytes,
                        energy=energy)

def make_policy(name: str,
                seq_length: int,
                num_features: int,
                encryption_mode: str,
                collect_mode: str,
                collection_rate: float,
                dataset: str,
                should_compress: bool,
                **kwargs: Dict[str, Any]) -> Policy:
    name = name.lower()

    # Look up the data-specific precision and width
    base = os.path.dirname(__file__)
    quantize_path = os.path.join(base, 'datasets', dataset, 'quantize.json')

    quantize_dict = read_json(quantize_path)
    precision = quantize_dict['precision']
    width = quantize_dict['width']
    max_skip = quantize_dict.get('max_skip', 1)
    use_min_skip = quantize_dict.get('use_min_skip', False)
    threshold_factor = quantize_dict.get('threshold_factor', 1.0)

    if name == 'random':
        return RandomPolicy(collection_rate=collection_rate,
                            precision=precision,
                            width=width,
                            num_features=num_features,
                            seq_length=seq_length,
                            encryption_mode=EncryptionMode[encryption_mode.upper()],
                            collect_mode=CollectMode[collect_mode.upper()],
                            should_compress=should_compress)
    elif name == 'uniform':
        return UniformPolicy(collection_rate=collection_rate,
                             precision=precision,
                             width=width,
                             num_features=num_features,
                             seq_length=seq_length,
                             encryption_mode=EncryptionMode[encryption_mode.upper()],
                             collect_mode=CollectMode[collect_mode.upper()],
                             should_compress=should_compress)
    elif name.startswith('adaptive'):
        # Look up the threshold path
        threshold_path = os.path.join(base, 'saved_models', dataset, 'thresholds.json.gz')

        did_find_threshold = False

        if not os.path.exists(threshold_path):
            print('WARNING: No threshold path exists.')
            threshold = 0.0
        else:
            thresholds = read_json_gz(threshold_path)
            rate_str = str(round(collection_rate, 2))

            if (name not in thresholds) or (collect_mode not in thresholds[name]) or (rate_str not in thresholds[name][collect_mode]):
                print('WARNING: No threshold path exists.')
                threshold = 0.0
            else:
                threshold = thresholds[name][collect_mode][rate_str]
                did_find_threshold = True

        # Apply the optional data-specific threshold factor
        if isinstance(threshold_factor, OrderedDict):
            rate_str = str(round(collection_rate, 2))

            if rate_str in threshold_factor:
                threshold *= threshold_factor[rate_str]
        else:
            threshold *= threshold_factor

        # Get the optional max skip value
        if isinstance(max_skip, OrderedDict):
            rate_str = str(round(collection_rate, 2))
            max_skip_value = max_skip.get(rate_str, max_skip['default'])
        else:
            max_skip_value = max_skip

        encoding_mode = str(kwargs['encoding']).lower()

        # For 'padded' policies, read the standard test log (if exists) to get the maximum number of collected values.
        # This is an impractical policy to use, as it requires prior knowledge of what the policy will do on the test
        # set. Nevertheless, we use this strategy to provide an 'ideal' baseline.
        max_collected = None

        if encoding_mode == 'padded':
            rate_str = str(int(round(collection_rate, 2) * 100))
            policy_name = '{0}_{1}'.format(name, encoding_mode)
            file_name = '{0}-{1}-{2}-{3}_{4}.json.gz'.format(name, encoding_mode, encryption_mode.lower(), collect_mode.lower(), rate_str)
            standard_path = os.path.join(base, 'saved_models', dataset, collect_mode.lower(), policy_name, file_name)

            sim_log = read_json_gz(standard_path)
            max_collected = max(sim_log['num_measurements'])

        if name == 'adaptive_heuristic':
            cls = AdaptiveHeuristic
        elif name == 'adaptive_litesense':
            cls = AdaptiveLiteSense
        elif name == 'adaptive_deviation':
            cls = AdaptiveDeviation
        else:
            raise ValueError('Unknown adaptive policy with name: {0}'.format(name))

        return cls(collection_rate=collection_rate,
                   threshold=threshold,
                   precision=precision,
                   width=width,
                   seq_length=seq_length,
                   num_features=num_features,
                   max_skip=max_skip_value,
                   use_min_skip=use_min_skip,
                   encryption_mode=EncryptionMode[encryption_mode.upper()],
                   collect_mode=CollectMode[collect_mode.upper()],
                   encoding_mode=EncodingMode[encoding_mode.upper()],
                   should_compress=should_compress,
                   max_collected=max_collected)
    elif (name == 'skip_rnn'):
        return SkipRNN(collection_rate=collection_rate,
                       threshold=0.5,
                       precision=precision,
                       width=width,
                       seq_length=seq_length,
                       num_features=num_features,
                       dataset_name=dataset,
                       encryption_mode=EncryptionMode[encryption_mode.upper()],
                       collect_mode=CollectMode[collect_mode.upper()],
                       encoding_mode=EncodingMode[str(kwargs['encoding']).upper()],
                       should_compress=should_compress)
    else:
        raise ValueError('Unknown policy with name: {0}'.format(name))
