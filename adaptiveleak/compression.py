import numpy as np
import math
from collections import deque
from scipy import integrate
from typing import Any, Dict

from utils.data_utils import round_to_block, truncate_to_block, calculate_bytes
from utils.data_utils import calculate_grouped_bytes, get_num_groups
from utils.constants import AES_BLOCK_SIZE, BIG_NUMBER


MAX_ITER = 25


class AdaptiveWidth:

    def __init__(self, target_frac: float, seq_length: int, num_features: int, width: int):
        self._target_frac = target_frac
        self._seq_length = seq_length
        self._num_features = num_features
        self._width = width

        # Account for the need to send the bit-width along
        # with the features
        self._target_bytes = target_frac * seq_length * num_features - 1
        self._rand = np.random.RandomState(seed=4850)

    @property
    def target_bytes(self) -> float:
        return self._target_bytes

    @property
    def target_bits(self) -> float:
        return 8 * self.target_bytes

    @property
    def group_size(self) -> int:
        return self._seq_length

    def __str__(self) -> str:
        return 'Adaptive Width'

    def get_width(self, num_transmitted: int, should_pad: bool) -> int:
        raise NotImplementedError()


class FixedWidth(AdaptiveWidth):

    def get_width(self, num_transmitted: int, should_pad: bool) -> int:
        return self._width

    def __str__(self) -> str:
        return 'Fixed'


class StableWidth(AdaptiveWidth):

    def get_width(self, num_transmitted: int, should_pad: bool) -> int:
        width = 8 * self._target_frac
        total_bytes = calculate_bytes(width=self._width,
                                      num_transmitted=num_transmitted,
                                      num_features=self._num_features,
                                      should_pad=should_pad)
        return int(round(width)), total_bytes

    def __str__(self) -> str:
        return 'Stable'


class StochasticBlockWidth(AdaptiveWidth):

    def get_width(self, num_transmitted: int, should_pad: bool) -> int:
        upper_bytes = round_to_block(self.target_bytes, AES_BLOCK_SIZE)
        lower_bytes = truncate_to_block(self.target_bytes, AES_BLOCK_SIZE)

        num_transmitted_features = num_transmitted * self._num_features

        # Get the adaptive width using stochastic rounding
        adaptive_width = int(self.target_bits / num_transmitted_features)

        to_bytes = lambda w: (1 + ((w * num_transmitted_features) / 8))
        recovered_bytes = to_bytes(adaptive_width)

        r = self._rand.uniform()

        if r < 0.5:
            # Round up to the upper bytes threshold
            while to_bytes(adaptive_width) <= upper_bytes:
                adaptive_width += 1

            adaptive_width -= 1
        else:
            # Round down to the lower bytes threshold
            while to_bytes(adaptive_width) > lower_bytes:
                adaptive_width -= 1

        return adaptive_width

    def __str__(self) -> str:
        return 'Stochastic Block'


class BlockWidth(AdaptiveWidth):

    @property
    def group_size(self) -> int:
        return max(int(math.floor((8 * AES_BLOCK_SIZE) / self._num_features)), 1)

    def __str__(self) -> str:
        return 'Block'

    def get_width(self, num_transmitted: int, should_pad: bool) -> int:
        
        # The total number of bits if we were to send everything (at cost)

        if should_pad:
            proj_transmitted = int(self._target_frac * self._seq_length)
            stable_bytes = calculate_bytes(width=self._width,
                                           num_transmitted=proj_transmitted,
                                           num_features=self._num_features,
                                           should_pad=should_pad)
        else:
            stable_bytes = int(self.target_bytes)

        # Pick the largest width to fit in the block
        num_groups = get_num_groups(num_transmitted=num_transmitted,
                                    group_size=self.group_size)

        start_width = int(math.ceil(self.target_bits / (self._num_features * num_transmitted)))

        widths: List[int] = [start_width for _ in range(num_groups)]

        data_bytes = calculate_grouped_bytes(widths=widths,
                                             num_transmitted=num_transmitted,
                                             num_features=self._num_features,
                                             group_size=self.group_size,
                                             should_pad=should_pad)

        group_idx = 0

        best_widths = [start_width for _ in range(num_groups)]
        best_error = BIG_NUMBER

        if should_pad:
            i = 0
            while (i < MAX_ITER) and (data_bytes != stable_bytes):
                if (data_bytes <= stable_bytes):
                    widths[group_idx] += 1
                else:
                    widths[group_idx] -= 1

                data_bytes = calculate_grouped_bytes(widths=widths,
                                                     num_transmitted=num_transmitted,
                                                     num_features=self._num_features,
                                                     group_size=self.group_size,
                                                     should_pad=should_pad)
                i += 1
                group_idx += 1
                group_idx = group_idx % len(widths)
        else:
            i = 0
            while (i < MAX_ITER) and (data_bytes > stable_bytes):
                widths[group_idx] -= 1
                data_bytes = calculate_grouped_bytes(widths=widths,
                                                     num_transmitted=num_transmitted,
                                                     num_features=self._num_features,
                                                     group_size=self.group_size,
                                                     should_pad=should_pad)
                i += 1
                group_idx += 1
                group_idx = group_idx % len(widths)

        # Should never hit this
        # assert data_bytes == stable_bytes, 'Transmitted: {0}, Widths: {1}, Data Bytes: {2}, Max Bytes: {3}'.format(num_transmitted, widths, data_bytes, max_bytes)

        return widths


class PIDWidth(AdaptiveWidth):

    def __init__(self,
                 target_frac: float,
                 seq_length: int,
                 num_features: int,
                 width: int,
                 kp: float,
                 ki: float,
                 kd: float):
        super().__init__(target_frac=target_frac,
                         seq_length=seq_length,
                         num_features=num_features,
                         width=width)
        self._kp = kp
        self._ki = ki
        self._kd = kd

        self._errors: deque = deque()
        self._offset = 0

    def get_width(self, num_transmitted: int, should_pad: bool) -> int:
       
        base_width = int(round(self.target_bits / (self._num_features * num_transmitted)))

        pred_bytes = calculate_bytes(width=base_width + self._offset,
                                     num_features=self._num_features,
                                     num_transmitted=num_transmitted,
                                     should_pad=should_pad)

        # Positive if too high, Negative if too low
        error = pred_bytes - self.target_bytes
        
        self._errors.append(error)
        while len(self._errors) > 100:
            self._errors.popleft()

        error_derivative = (self._errors[-1] - self._errors[-2]) if len(self._errors) >= 2 else 0.0

        prop_term = self._kp * error
        integral_term = self._ki * integrate.trapz(self._errors)
        derivative_term = self._kd * error_derivative

        # Positive if too high, Negative if too low
        control_signal = prop_term + integral_term + derivative_term

        self._offset = int(self._offset - control_signal)
        self._offset = max(min(self._offset, 4), -4)

        return base_width + self._offset

    def __str__(self) -> str:
        return 'PID Width'


def make_compression(name: str,
                     num_features: int,
                     seq_length: int,
                     width: int,
                     target_frac: float,
                     **kwargs: Dict[str, Any]) -> AdaptiveWidth:
    name = name.lower()

    if name == 'fixed':
        return FixedWidth(num_features=num_features,
                          seq_length=seq_length,
                          target_frac=target_frac,
                          width=width)
    elif name == 'stable':
        return StableWidth(num_features=num_features,
                           seq_length=seq_length,
                           target_frac=target_frac,
                           width=width)
    elif name == 'stochastic_block':
        return StochasticBlockWidth(num_features=num_features,
                                    seq_length=seq_length,
                                    target_frac=target_frac,
                                    width=width)
    elif name == 'block':
        return BlockWidth(num_features=num_features,
                          seq_length=seq_length,
                          target_frac=target_frac,
                          width=width)
    elif name == 'pid':
        return PIDWidth(num_features=num_features,
                        seq_length=seq_length,
                        width=width,
                        target_frac=target_frac,
                        kp=kwargs['kp'],
                        ki=kwargs['ki'],
                        kd=kwargs['kd'])
    else:
        raise ValueError('Unknown compression name: {0}'.format(name))
