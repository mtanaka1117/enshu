import numpy as np
import math
from functools import partial
from Cryptodome.Random import get_random_bytes
from typing import List

from adaptiveleak.utils.constants import BIT_WIDTH
from adaptiveleak.utils.encryption import AES_BLOCK_SIZE, EncryptionMode, CHACHA_NONCE_LEN


MAX_ITER = 100


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
    min_val = -max_val

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


def linear_extrapolate(prev: np.ndarray, curr: np.ndarray, delta: float) -> np.ndarray:
    """
    This function uses a linear approximation over the given readings to project
    the next value 'delta' units ahead.

    Args:
        prev: A [D] array containing the previous value
        curr: A [D] array containing the current value
        delta: The time between consecutive readings
    Returns:
        A [D] array containing the projected reading (`delta` steps ahead).
    """
    slope = (curr - prev) / delta
    return slope * (2 * delta) + prev


def pad_to_length(message: bytes, length: int) -> bytes:
    """
    Pads larger messages to the given length.
    """
    if len(message) >= length:
        return message

    padding = get_random_bytes(length - len(message))
    return message + padding


def round_to_block(x: float, block_size: int) -> int:
    return int(math.ceil(x / block_size)) * block_size


def truncate_to_block(x: float, block_size: int) -> int:
    return int(math.floor(x / block_size)) * block_size


def pack(values: List[int], width: int) -> bytes:
    """
    Packs the list of (quantized) values with the given width
    into a packed bit-string.

    Args:
        values: The list of quantized values
        width: The width of each quantized value
    Returns:
        A packed string containing the quantized values.
    """
    packed: List[int] = [0]
    consumed = 0
    num_bytes = int(math.ceil(width / 8))

    for value in values:
        for i in range(num_bytes):
            # Get the current byte
            current_byte = (value >> (i * 8)) & 0xFF

            # Get the number of used bits in the current byte
            if i < (num_bytes - 1) or (width % 8) == 0:
                num_bits = 8
            else:
                num_bits = width % 8

            # Set bits in the packed string
            packed[-1] |= current_byte << consumed
            packed[-1] &= 0xFF

            # Add to the number of consumed bits
            used_bits = min(8 - consumed, num_bits)
            consumed += num_bits

            # If we have consumed more than a byte, then the remaining amount
            # spills onto the next byte
            if consumed > 8:
                consumed = consumed - 8
                remaining_value = current_byte >> used_bits

                # Add the remaining value to the running string
                packed.append(remaining_value)

    return bytes(packed)


def unpack(encoded: bytes, width: int,  num_values: int) -> List[int]:
    """
    Unpacks the encoded values into a list of integers of the given bit-width.

    Args:
        encoded: The encoded list of values (output of pack())
        width: The bit width for each value
        num_value: The number of encoded values
    Returns:
        A list of integer values
    """
    result: List[int] = []
    current = 0
    current_length = 0
    byte_idx = 0
    mask = (1 << width) - 1

    for i in range(num_values):
        # Get at at least the next 'width' bits
        while (current_length < width):
            current |= (encoded[byte_idx] << current_length)
            current_length += 8
            byte_idx += 1

        # Truncate down to 'width' bits
        value = current & mask
        result.append(value)

        current = current >> width
        current_length = current_length - width

    # Include any residual values
    if len(result) < num_values:
        result.append(current)

    return result


def get_group_widths(group_size: int,
                     num_collected: int,
                     num_features: int,
                     seq_length: int,
                     target_frac: float,
                     encryption_mode: EncryptionMode) -> List[int]:
    """
    Calculates the bit-width of each group such that the final encrypted message
    has the same length as the target value.

    Args:
        group_size: The number of measurements per group
        num_collected: The number of collected measurements
        num_features: The number of features per measurement
        seq_length: The number of elements in a full sequence
        target_frac: The target collection / sending fraction (on average)
        encryption_mode: The type of encryption algorithm (block or stream)
    Returns:
        A list of bit-widths for each group.
    """
    # Get the target values
    target_data_bits = BIT_WIDTH * num_collected * num_features

    target_bytes = calculate_bytes(width=BIT_WIDTH,
                                   num_collected=int(target_frac * seq_length),
                                   num_features=num_features,
                                   seq_length=seq_length,
                                   encryption_mode=encryption_mode)

    # Get the number of groups
    num_groups = get_num_groups(num_collected=num_collected,
                                group_size=group_size)

    # Pick the larger initial widths (add a fudge constant to ensure we over-estimate)
    start_width = int(math.ceil(target_data_bits / (num_features * num_collected)))
    widths = [start_width for _ in range(num_groups)]

    # Calculate the number of bytes with the initial widths
    data_bytes = calculate_grouped_bytes(widths=widths,
                                         num_collected=num_collected,
                                         num_features=num_features,
                                         group_size=group_size,
                                         seq_length=seq_length,
                                         encryption_mode=encryption_mode)

    # Set the group widths in a round-robin fashion
    i = 0
    group_idx = 0

    # if encryption_mode == EncryptionMode.BLOCK:
    while (i < MAX_ITER):
        # Adjust the group width
        widths[group_idx] += 1 if data_bytes <= target_bytes else -1

        # Update the byte count
        updated_bytes = calculate_grouped_bytes(widths=widths,
                                                num_collected=num_collected,
                                                num_features=num_features,
                                                group_size=group_size,
                                                seq_length=seq_length,
                                                encryption_mode=encryption_mode)

        # Exit the loop when we find the inflection point
        if (data_bytes <= target_bytes and updated_bytes > target_bytes):
            widths[group_idx] -= 1
            break

        data_bytes = updated_bytes
        i += 1
        group_idx += 1
        group_idx = group_idx % len(widths)

    return widths


def calculate_bytes(width: int, num_collected: int, num_features: int, seq_length: int, encryption_mode: EncryptionMode) -> int:
    """
    Calculates the number of bytes required to send the given
    number of features and measurements of the provided width.

    Args:
        width: The bit-width of each features
        num_collected: The number of collected measurements
        num_features: The number of features per measurement
        seq_length: The length of a full sequence
        encryption_mode: The type of encryption (block or stream)
    Returns:
        The total number of bytes in the encrypted message.
    """
    data_bits = width * num_collected * num_features
    data_bytes = int(math.ceil(data_bits / 8))

    # Include the meta-data (precision) and the sequence bit mask
    message_bytes = data_bytes + 1 + int(math.ceil(seq_length / 8))

    if encryption_mode == EncryptionMode.BLOCK:
        # Account for the IV
        return round_to_block(message_bytes + AES_BLOCK_SIZE, block_size=AES_BLOCK_SIZE)
    elif encryption_mode == EncryptionMode.STREAM:
        # Account for the nonce
        return message_bytes + CHACHA_NONCE_LEN
    else:
        raise ValueError('Unknown encryption mode: {0}'.format(encryption_mode.name))


def get_num_groups(num_collected: int, group_size: int) -> int:
    return int(math.ceil(num_collected / group_size))


def calculate_grouped_bytes(widths: List[int],
                            num_collected: int,
                            num_features: int,
                            group_size: int,
                            encryption_mode: EncryptionMode,
                            seq_length: int) -> int:
    """
    Calculates the number of bytes required to encode the given groups of features.

    Args:
        widths: A list of the bit-widths for each group
        num_collected: The number of collected measurements
        num_features: The number of features per measurement
        group_size: The number of measurements per group
        encryption_mode: The type of encryption algorithm (block or stream)
        seq_length: The length of a full sequence
    Returns:
        The number of bytes in the encoded message.
    """
    # Validate arguments
    num_groups = get_num_groups(num_collected=num_collected, group_size=group_size)
    assert len(widths) == num_groups, 'Must provide {0} widths. Got: {1}'.format(num_groups, len(widths))

    total_bytes = 0
    so_far = 0

    # Calculate the number of data bytes in the encoded message
    for idx, width in enumerate(widths):
        group_elements = min(group_size, num_collected - so_far)

        data_bits = width * group_elements * num_features
        data_bytes = int(math.ceil(data_bits / 8))

        total_bytes += data_bytes
        so_far += group_elements

    # Include the meta-data (group widths) and the sequence mask
    total_bytes += len(widths) + 2 + int(math.ceil(seq_length / 8))

    if encryption_mode == EncryptionMode.BLOCK:
        # Include the IV
        return AES_BLOCK_SIZE + round_to_block(total_bytes, AES_BLOCK_SIZE)
    elif encryption_mode == EncryptionMode.STREAM:
        # Include the Nonce
        return CHACHA_NONCE_LEN + total_bytes
    else:
        raise ValueError('Unknown encryption mode: {0}'.format(encryption_mode.name))
