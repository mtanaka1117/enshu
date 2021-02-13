import numpy as np
import math
from functools import partial

from utils.constants import AES_BLOCK_SIZE


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


def calculate_bytes(width: int, num_transmitted: int, num_features: int) -> int:
    data_bits = width * num_transmitted * num_features
    data_bytes = int(math.ceil(data_bits / 8)) + 1  # Account for the need to send along the width
    total_bytes = round_to_block(data_bytes, AES_BLOCK_SIZE)
    return total_bytes
