import numpy as np
import math
from functools import partial
from typing import List

from utils.constants import AES_BLOCK_SIZE


def apply_dropout(mat: np.ndarray, drop_rate: float, rand: np.random.RandomState) -> np.ndarray:
    rand_mat = rand.uniform(low=0.0, high=1.0, size=mat.shape)
    mask = np.less(rand_mat, drop_rate).astype(float)

    scale = 1.0 / (1.0 - drop_rate)
    scaled_mask = mask * scale
    
    return mat * scaled_mask


def leaky_relu(x: np.ndarray, alpha: float) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    max_x = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def to_fixed_point(x: float, precision: int, width: int) -> int:
    multiplier = (1 << precision)
    fp = int(x * multiplier)
    
    max_val = (1 << (width - 1)) - 1
    min_val = -((1 << (width - 1)) - 1)

    if fp > max_val:
        return max_val
    elif fp < min_val:
        return min_val
    return fp


def to_float(fp: int, precision: int) -> float:
    multiplier = float(1 << precision)
    return float(fp) / multiplier


def array_to_fp(arr: np.ndarray, precision: int, width: int) -> np.ndarray:
    convert_fn = partial(to_fixed_point, precision=precision, width=width)
    map_fn = np.vectorize(convert_fn)
    return map_fn(arr)


def array_to_float(fp_arr: np.ndarray, precision: int) -> np.ndarray:
    convert_fn = partial(to_float, precision=precision)
    map_fn = np.vectorize(convert_fn)

    return map_fn(fp_arr)


def round_to_block(x: float, block_size: int) -> int:
    return int(math.ceil(x / block_size)) * block_size


def truncate_to_block(x: float, block_size: int) -> int:
    return int(math.floor(x / block_size)) * block_size


def calculate_bytes(width: int, num_transmitted: int, num_features: int, should_pad: bool) -> int:
    data_bits = width * num_transmitted * num_features
    data_bytes = int(math.ceil(data_bits / 8)) + 1  # Account for the need to send along the width

    if should_pad:
        return round_to_block(data_bytes, AES_BLOCK_SIZE)

    return data_bytes


def get_num_groups(num_transmitted: int, group_size: int) -> int:
    return int(math.ceil(num_transmitted / group_size))


def calculate_grouped_bytes(widths: List[int], num_transmitted: int, num_features: int, group_size: int, should_pad: bool) -> int:
    # Validate arguments
    num_groups = get_num_groups(num_transmitted=num_transmitted, group_size=group_size)
    assert len(widths) == num_groups, 'Must provide {0} widths. Got: {1}'.format(num_groups, len(widths))

    total_bytes = 0
    so_far = 0

    for idx, width in enumerate(widths):
        group_elements = min(group_size, num_transmitted - so_far)

        total_bytes += calculate_bytes(width=width,
                                       num_transmitted=group_elements,
                                       num_features=num_features,
                                       should_pad=should_pad)
        so_far += group_elements

    return total_bytes
