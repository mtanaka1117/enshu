import numpy as np
import math


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
    assert len(arr.shape) == 1, 'Can only map 1d arrays to fixed-point representation'
    return np.array([to_fixed_point(x, precision=precision, width=width) for x in arr])


def array_to_float(fp_arr: np.ndarray, precision: int) -> np.ndarray:
    assert len(fp_arr.shape) == 1, 'Can only map 1d arrays to floating point representation'
    return np.array([to_float(fp, precision=precision) for fp in fp_arr])


def round_to_block(x: float, block_size: int) -> int:
    return int(math.ceil(x / block_size)) * block_size


def truncate_to_block(x: float, block_size: int) -> int:
    return int(math.floor(x / block_size)) * block_size
