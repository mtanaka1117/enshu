import numpy as np
import math
import time
from functools import partial
from Cryptodome.Random import get_random_bytes
from typing import List, Union, Tuple, Iterable

from adaptiveleak.utils.constants import BITS_PER_BYTE, BIG_NUMBER, MIN_WIDTH, SMALL_NUMBER, BOUND_BITS, MAX_WIDTH
from adaptiveleak.utils.encryption import AES_BLOCK_SIZE, EncryptionMode, CHACHA_NONCE_LEN


MAX_ITER = 100
ERROR_TOL = 0


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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-1 * x))


def linear_sigmoid(x: np.ndarray) -> np.ndarray:
    cond1 = (x < -3).astype(float)
    cond2 = np.logical_and(x >= -3, x < -1).astype(float)
    cond3 = np.logical_and(x >= -1, x < 1).astype(float)
    cond4 = np.logical_and(x >= 1, x <= 3).astype(float)
    cond5 = (x > 3).astype(float)

    part1 = 0
    part2 = 0.125 * x + 0.375
    part3 = 0.25 * x + 0.5
    part4 = 0.125 * x + 0.625
    part5 = 1

    return (part1 * cond1) + (part2 * cond2) + (part3 * cond3) + (part4 * cond4) + (part5 * cond5)


def linear_tanh(x: np.ndarray) -> np.ndarray:
    return 2 * linear_sigmoid(2 * x) - 1


def to_fixed_point(x: float, precision: int, width: int) -> int:
    assert width >= 1, 'Must have a non-negative width'

    multiplier = 1 << abs(precision)
    fp = int(round(x * multiplier)) if precision > 0 else int(round(x / multiplier))

    max_val = (1 << (width - 1)) - 1
    min_val = -max_val

    if fp > max_val:
        return max_val
    elif fp < min_val:
        return min_val
    return fp


def to_float(fp: int, precision: int) -> float:
    multiplier = float(1 << abs(precision))
    return float(fp) / multiplier if precision > 0 else float(fp) * multiplier


def precision_change_error(fixed_point: int, old_precision: int, new_precision: int) -> float:
    """
    Calculates the error induced by changing the given fixed point value to a new precision.

    Args:
        fixed_point: The current fixed point value
        old_precision: The old precision value
        new_precision: The new precision value
    Returns:
        The error (pos if worse, neg if better) when changing to the new precision.
    """
    return to_float(fixed_point, old_precision) - to_float(fixed_point, new_precision)


def precision_change_error_array(fixed_point: np.ndarray, old_precision: int, new_precision: int) -> float:
    """
    Calculates the error induced by changing the given fixed point value to a new precision.

    Args:
        fixed_point: The current fixed point value
        old_precision: The old precision value
        new_precision: The new precision value
    Returns:
        The error (pos if worse, neg if better) when changing to the new precision.
    """
    return np.sum(array_to_float(fixed_point, old_precision) - array_to_float(fixed_point, new_precision))


def array_to_fp(arr: np.ndarray, precision: int, width: int) -> np.ndarray:
    multiplier = 1 << abs(precision)
    
    if precision > 0:
        quantized = arr * multiplier
    else:
        quantized = arr / multiplier

    quantized = np.round(quantized).astype(int)

    max_val = (1 << (width - 1)) - 1
    min_val = -max_val

    return np.clip(quantized, a_min=min_val, a_max=max_val)


def array_to_fp_unsigned(arr: np.ndarray, precision: int, width: int) -> np.ndarray:
    multiplier = 1 << abs(precision)

    if (precision > 0):
        quantized = arr * multiplier
    else:
        quantized = arr / multiplier

    quantized = np.round(quantized).astype(int)

    max_val = (1 << width) - 1
    min_val = 0

    return np.clip(quantized, a_min=min_val, a_max=max_val)


def array_to_fp_shifted(arr: np.ndarray, precision: int, width: int, shifts: np.ndarray) -> np.ndarray:
    assert len(arr.shape) == 1, 'Must provide a 1d array'
    assert arr.shape == shifts.shape, 'Misaligned data {0} and shifts {1}'.format(arr.shape, shifts.shape)

    shifted_precisions = precision - shifts
    multipliers = np.left_shift(np.ones_like(shifts), np.abs(shifted_precisions))

    quantized = np.where(shifted_precisions > 0, arr * multipliers, arr / multipliers)
    quantized = np.round(quantized).astype(int)

    max_val = (1 << (width - 1)) - 1
    min_val = -max_val

    return np.clip(quantized, a_min=min_val, a_max=max_val)


def array_to_float(fp_arr: Union[np.ndarray, List[int]], precision: int) -> np.ndarray:
    multiplier = float(1 << abs(precision))

    if isinstance(fp_arr, list):
        fp_arr = np.array(fp_arr)

    if precision > 0:
        return fp_arr.astype(float) / multiplier
    else:
        return fp_arr.astype(float) * multiplier


def array_to_float_shifted(arr: Union[np.ndarray, List[int]], precision: int, shifts: np.ndarray) -> np.ndarray:
    shifted_precisions = precision - shifts
    multipliers = np.left_shift(np.ones_like(shifts), np.abs(shifted_precisions))

    if isinstance(arr, list):
        fp_arr = np.array(arr).astype(float)
    else:
        fp_arr = arr.astype(float)

    assert len(fp_arr.shape) == 1, 'Must provide a 1d array'
    assert fp_arr.shape == shifts.shape, 'Misaligned data {0} and shifts {1}'.format(fp_arr.shape, shifts.shape)

    recovered = np.where(shifted_precisions > 0, fp_arr / multipliers, fp_arr * multipliers)

    return recovered


def select_range_shift(measurement: int, old_width: int, old_precision: int, new_width: int, num_range_bits: int, prev_shift: int) -> int:
    """
    Selects the lowest-error range multiplier.

    Args:
        measurement: The fixed point measurement feature
        old_width: The standard width of each feature
        old_precision: The existing precision of each feature
        new_width: The quantized width of each feature (this is how we will express values)
        num_range_bits: The number of bits for the range exponent
    Returns:
        The range exponent in [-2^{range_bits - 1}, 2^{range_bits - 1}]
    """
    assert num_range_bits >= 1, 'Number of range bits must be non-negative'
    assert (old_width >= 1) and (new_width >= 1), 'Number of width bits must be non-negative'

    # Create the constants necessary for selecting the range shift
    non_fractional = old_width - old_precision
    width_mask = (1 << (new_width - 1)) - 1  # Masks out all non-data bits (including the sign bit)
    recovered_mask = 0x7FFF

    # Get the new precision based on the (fixed) number of non-fractional bits 
    #new_precision = new_width - non_fractional
    base_shift = old_width - new_width

    # Get the absolute value of the given fixed point measurement. The routine
    # performs all operations on positive values for simplicity.
    abs_value = abs(measurement)

    # Try the previous shift first for potential early-exiting with a greedy algorithm
    conversion_shift = base_shift + prev_shift

    if (conversion_shift >= 0):
        quantized = (abs_value >> conversion_shift) & width_mask
        recovered = (quantized << conversion_shift)
    else:
        conversion_shift *= -1
        quantized = (abs_value << conversion_shift) & width_mask
        recovered = (quantized >> conversion_shift)

    recovered &= recovered_mask

    prev_error = abs(abs_value - recovered)

    if (prev_error <= ERROR_TOL):
        return prev_shift

    last_error = BIG_NUMBER
    best_error = prev_error
    best_shift = prev_shift

    offset = (1 << (num_range_bits - 1))
    limit = 1 << num_range_bits

    for idx in range(limit):
        shift = idx - offset
        
        # Convert the value to and from floating point
        conversion_shift = base_shift + shift

        if (conversion_shift >= 0):
            quantized = (abs_value >> conversion_shift) & width_mask
            recovered = (quantized << conversion_shift)
        else:
            conversion_shift *= -1
            quantized = (abs_value << conversion_shift) & width_mask
            recovered = (quantized >> conversion_shift)

        recovered &= recovered_mask

        # Compute the error and save the best result
        error = abs(abs_value - recovered)

        if (error < best_error):
            best_shift = shift
            best_error = error

        if (error > last_error) or (best_error == 0):
            break  # Stop once the error starts to increase

        last_error = error

    if (prev_error <= best_error):
        return prev_shift

    return best_shift


def select_range_shifts_array(measurements: np.ndarray, old_width: int, old_precision: int, new_width: int, num_range_bits: int) -> np.ndarray:
    """
    Selects the lowest-error range multiplier.

    Args:
        measurements: A 1d array of measurement features
        old_width: The standard width of existing features
        old_precision: The precision of existing features
        new_width: The new width of each feature (this is how values will be encoded)
        precision: The precision of each feature
        num_range_bits: The number of bits for the range exponent
    Returns:
        The range exponent in [-2^{range_bits - 1}, 2^{range_bits - 1}]
    """
    assert num_range_bits >= 1, 'Number of range bits must be non-negative'
    assert (old_width >= 1) and (new_width >= 1), 'Number of width bits must be non-negative'
    assert len(measurements.shape) == 1, 'Must provide a 1d numpy array'

    # Convert all values to fixed point
    fp_values = array_to_fp(measurements, width=old_width, precision=old_precision)

    num_values = measurements.shape[0]
    best_shifts = np.empty(num_values)
    prev_shift = 0

    elapsed = []

    for idx in range(num_values):
        shift = select_range_shift(measurement=fp_values[idx],
                                   old_width=old_width,
                                   old_precision=old_precision,
                                   new_width=new_width,
                                   num_range_bits=num_range_bits,
                                   prev_shift=prev_shift)

        best_shifts[idx] = int(shift)
        prev_shift = shift

    return best_shifts.astype(int)

    #abs_values = np.abs(measurements)
    #max_representable_fp = (1 << (width - 1)) - 1
    #non_fractional = width - precision

    #best_errors = np.ones_like(abs_values) * BIG_NUMBER
    #best_shifts = np.ones_like(abs_values) * ((1 << (num_range_bits - 1)) - 1)

    #offset = 1 << (num_range_bits - 1)

    #for idx in range(pow(2, num_range_bits)):
    #    shift = idx - offset
    #    
    #    shifted_max = to_float(max_representable_fp, precision=precision - shift)
    #    errors = np.abs(shifted_max - abs_values)

    #    cond = np.logical_and(errors < best_errors, shifted_max >= abs_values)
    #    best_errors = np.where(cond, errors, best_errors)
    #    best_shifts = np.where(cond, shift, best_shifts)

    #return best_shifts.astype(int)


def linear_extrapolate(prev: np.ndarray, curr: np.ndarray, delta: float, num_steps: int) -> np.ndarray:
    """
    This function uses a linear approximation over the given readings to project
    the next value 'delta' units ahead.

    Args:
        prev: A [D] array containing the previous value
        curr: A [D] array containing the current value
        delta: The time between consecutive readings
        num_steps: The number of steps to extrapolate ahead
    Returns:
        A [D] array containing the projected reading (`num_steps * delta` steps ahead).
    """
    slope = (curr - prev) / delta
    return slope * delta * num_steps + curr


def pad_to_length(message: bytes, length: int) -> bytes:
    """
    Pads larger messages to the given length by appending
    random bytes.
    """
    if len(message) >= length:
        return message

    padding = get_random_bytes(length - len(message))
    return message + padding


def round_to_block(length: Union[int, float], block_size: int) -> int:
    """
    Rounds the given length to the nearest (larger) multiple of
    the block size.
    """
    if isinstance(length, int) and (length % block_size == 0):
        return length

    return int(math.ceil(length / block_size)) * block_size


def truncate_to_block(length: Union[int, float], block_size: int) -> int:
    """
    Rounds the given length to the nearest (smaller) multiple of
    the block size.
    """
    return int(math.floor(length / block_size)) * block_size


def get_max_collected(seq_length: int,
                      num_features: int,
                      group_size: int,
                      min_width: int,
                      target_size: int,
                      encryption_mode: EncryptionMode) -> int:
    """
    Returns the maximum number of measurements that can be quantized
    to the given target size with features of (at least) the minimum width.

    Args:
        seq_length: The length of a full sequence
        num_features: The number of features in a single measurement
        group_size: The number of features in each group except (possibly) the last
        min_width: The minimum bit width of a single feature
        target_size: The target number of bytes
        encryption_mode: The type of encryption algorithm (block or stream)
    Returns:
        The maximum number of measurements
    """
    # Get the number of bytes per group (in the worst case)
    size_per_group = int(math.ceil((group_size * min_width) / BITS_PER_BYTE))

    # Get the number of meta-data bytes
    meta_size = int(math.ceil(seq_length / BITS_PER_BYTE)) + 2
    if encryption_mode == EncryptionMode.BLOCK:
        meta_size += AES_BLOCK_SIZE
    elif encryption_mode == EncryptionMode.STREAM:
        meta_size += CHACHA_NONCE_LEN
    else:
        raise ValueError('Unknown encryption mode {0}'.format(encryption_mode.name))

    # Calculate the maximum number of groups
    max_groups = int(math.floor((target_size - meta_size) / (size_per_group + 1.0)))

    # Determine the number of consumed bytes thus far
    current_size = int(max_groups * size_per_group + max_groups + meta_size)

    # Get the maximum number of measurements based on
    # even-sized groups
    max_features = max_groups * group_size

    # Add in extra measurements to the final group (may be smaller than other groups)
    extra_features = int(math.floor((BITS_PER_BYTE * ((target_size - current_size) - 1)) / min_width))
    max_features += extra_features

    # Calculate the maximum number of full measurements
    max_measurements = int(math.floor(max_features / num_features))

    # Cap the number of measurements at the sequence length
    return min(max_measurements, seq_length)


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


def set_widths(group_sizes: List[int], target_bytes: int, start_width: int) -> List[int]:
    """
    Sets the group widths in a round-robin fashion
    to saturate the target bytes.

    Args:
        group_sizes: The size (in number of features) of each group
        target_bytes: The target number of data bytes
        start_width: The starting number of bits per feature
    Returns:
        A list of the bit widths for each group
    """
    num_groups = len(group_sizes)
    num_values = sum(group_sizes)
    target_bits = target_bytes * BITS_PER_BYTE

    # Set the initial widths based on an even distribution
    #even_width = int(target_bits / num_values)
    consumed_bytes = sum((int(math.ceil((start_width * size) / BITS_PER_BYTE)) for size in group_sizes))

    start_widths = min(start_width, MAX_WIDTH)
    widths: List[int] = [start_width for _ in range(num_groups)]

    if start_width >= MAX_WIDTH:
        return widths

    counter = 0
    has_improved = True
    while (has_improved and counter < MAX_ITER):

        has_improved = False

        for idx in range(len(group_sizes)):
            if (widths[idx] == MAX_WIDTH):
                continue

            # Increase this group's width by 1 bit
            widths[idx] += 1

            # Calculate the new number of data bytes
            candidate_bytes = sum((int(math.ceil((w * size) / BITS_PER_BYTE)) for w, size in zip(widths, group_sizes)))

            if (candidate_bytes <= target_bytes):
                consumed_bytes = candidate_bytes
                has_improved = True
            else:
                widths[idx] -= 1

        counter += 1

    return widths


def get_group_widths(group_size: int,
                     num_collected: int,
                     num_features: int,
                     seq_length: int,
                     target_frac: float,
                     standard_width: int,
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
        standard_width: The standard bit-width for each feature
        encryption_mode: The type of encryption algorithm (block or stream)
    Returns:
        A list of bit-widths for each group.
    """
    # Get the target values
    target_collected = int(target_frac * seq_length)
    target_data_bits = standard_width * target_collected * num_features

    target_bytes = calculate_bytes(width=standard_width,
                                   num_collected=target_collected,
                                   num_features=num_features,
                                   seq_length=seq_length,
                                   encryption_mode=encryption_mode)

    # Get the number of groups
    num_groups = get_num_groups(num_collected=num_collected,
                                num_features=num_features,
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

    while (i < MAX_ITER):
        # Adjust the group width
        widths[group_idx] += 1 if data_bytes <= target_bytes else -1
        widths[group_idx] = max(widths[group_idx], MIN_WIDTH)

        # Update the byte count
        updated_bytes = calculate_grouped_bytes(widths=widths,
                                                num_collected=num_collected,
                                                num_features=num_features,
                                                group_size=group_size,
                                                seq_length=seq_length,
                                                encryption_mode=encryption_mode)
    
        # Exit the loop when we find the inflection point
        if (data_bytes <= target_bytes and updated_bytes > target_bytes):
            widths[group_idx] = max(widths[group_idx] - 1, MIN_WIDTH)
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
        width: The bit-width of each feature
        num_collected: The number of collected measurements
        num_features: The number of features per measurement
        seq_length: The length of a full sequence
        encryption_mode: The type of encryption (block or stream)
    Returns:
        The total number of bytes in the encrypted message.
    """
    data_bits = width * num_collected * num_features
    data_bytes = int(math.ceil(data_bits / BITS_PER_BYTE))

    # Include the meta-data (the sequence bit mask)
    message_bytes = data_bytes + int(math.ceil(seq_length / BITS_PER_BYTE))

    if encryption_mode == EncryptionMode.BLOCK:
        # Account for the IV
        return round_to_block(message_bytes + AES_BLOCK_SIZE, block_size=AES_BLOCK_SIZE)
    elif encryption_mode == EncryptionMode.STREAM:
        # Account for the nonce
        return message_bytes + CHACHA_NONCE_LEN
    else:
        raise ValueError('Unknown encryption mode: {0}'.format(encryption_mode.name))


def get_num_groups(num_collected: int, num_features: int, group_size: int) -> int:
    return int(math.ceil((num_collected * num_features) / group_size))


def balance_group_size(num_collected: int, num_features: int, max_group_size: int) -> int:
    """
    Balances groups by evenly spreading the features across each grouping.

    Args:
        num_collected: The number of collected measurements (K)
        num_features: The number of features per measurement (D)
        max_group_size: The maximum number of features per group (G)
    Returns:
        The selected (balanced) group size in [1, G]
    """
    num_groups = get_num_groups(num_collected=num_collected,
                                num_features=num_features,
                                group_size=max_group_size)

    total_features = num_collected * num_features
    even_group_size = int(math.ceil(total_features / num_groups))

    return min(max(even_group_size, 1), max_group_size)


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
        group_size: The number of features per group
        encryption_mode: The type of encryption algorithm (block or stream)
        seq_length: The length of a full sequence
    Returns:
        The number of bytes in the encoded message.
    """
    # Validate arguments
    num_groups = get_num_groups(num_collected=num_collected,
                                group_size=group_size,
                                num_features=num_features)
    assert len(widths) == num_groups, 'Must provide {0} widths. Got: {1}'.format(num_groups, len(widths))

    total_bytes = 0
    so_far = 0
    total_features = num_collected * num_features

    # Calculate the number of data bytes in the encoded message
    for idx, width in enumerate(widths):
        group_elements = min(group_size, total_features - so_far)

        data_bits = width * group_elements
        data_bytes = int(math.ceil(data_bits / 8))

        total_bytes += data_bytes
        so_far += group_elements

    # Include the meta-data (group widths) and the sequence mask
    total_bytes += num_groups + int(math.ceil(seq_length / 8)) + 1

    if encryption_mode == EncryptionMode.BLOCK:
        # Include the IV
        return AES_BLOCK_SIZE + round_to_block(total_bytes, AES_BLOCK_SIZE)
    elif encryption_mode == EncryptionMode.STREAM:
        # Include the Nonce
        return CHACHA_NONCE_LEN + total_bytes
    else:
        raise ValueError('Unknown encryption mode: {0}'.format(encryption_mode.name))


def prune_sequence(measurements: np.ndarray, collected_indices: List[int], max_collected: int, seq_length: int) -> Tuple[np.ndarray, List[int]]:
    """
    Prunes the given sequence to use at most the maximum number
    of measurements. We remove measurements that induce the approximate lowest
    amount of additional error.

    Args:
        measurements: A [K, D] array of collected measurement vectors
        collected_indices: A list of [K] indices of the collected measurements
        max_collected: The maximum number of allowed measurements
        seq_length: The full sequence length
    Returns:
        A tuple of two elements:
            (1) A [K', D] array of the pruned measurements
            (2) A [K'] list of the remaining indices
    """
    assert len(measurements.shape) == 2, 'Must provide a 2d array of measurements'
    assert measurements.shape[0] == len(collected_indices), 'Misaligned measurements ({0}) and collected indices ({1})'.format(measurements.shape[0], len(collected_indices))

    # Avoid pruning measurements which are under budget
    num_collected = len(collected_indices)
    if num_collected <= max_collected:
        return measurements, collected_indices

    # Compute the differences in measurements
    first = measurements[:-1]  # [L - 1, D]
    last = measurements[1:]  # [L - 1, D]
    measurement_diff = np.sum(np.abs(last - first), axis=-1).astype(int)  # [L - 1]

    # Compute the two-step differences in indices
    idx_diff = np.array([(collected_indices[i+1] - collected_indices[i]) for i in range(len(collected_indices) - 1)])  # [L - 1]
    
    scores = measurement_diff + (0.125 * idx_diff)  # [L - 1]
    scores = scores[:-1]  # [L - 2]

    num_to_prune = len(measurements) - max_collected

    if num_to_prune >= len(scores):
        to_remove_set = set(i + 1 for i in range(len(scores)))
    else:
        idx_to_prune = np.argpartition(scores, num_to_prune)[0:num_to_prune]
        idx_to_prune += 1

        # Remove the given elements from the measurements and collected lists
        to_remove_set = set(idx_to_prune)
 
    idx_to_keep = [i for i in range(len(measurements)) if i not in to_remove_set]

    pruned_measurements = measurements[idx_to_keep]
    pruned_indices = [collected_indices[i] for i in idx_to_keep]

    return pruned_measurements, pruned_indices


def create_groups(measurements: np.ndarray, max_num_groups: int, max_group_size: int) -> List[np.ndarray]:
    """
    Creates measurement groups using a greedy algorithm based on similar signs.

    Args:
        measurements: A [K, D] array of measurements
        max_num_groups: The maximum number of groups (L)
        max_group_size: The maximum number of features in a group
    Returns:
        A list of 1d arrays of flattened measurements
    """
    assert len(measurements.shape) == 2, 'Must provide a 2d measurements array'

    # Flatten the features into a 1d array (feature-wise)
    flattened = measurements.T.reshape(-1)

    min_group_size = int(math.ceil(len(flattened) / max_num_groups))

    groups: List[np.ndarray] = []
    current_idx = 0
    current_size = min_group_size

    indices = np.arange(len(flattened))
    signs = (np.greater_equal(flattened, 0)).astype(float)

    while (current_idx < len(flattened)):
        end_idx = current_idx + current_size

        if (end_idx > len(flattened)):
            groups.append(flattened[current_idx:])
            break

        is_positive = np.all(flattened[current_idx:end_idx] > 0)
        is_negative = np.all(flattened[current_idx:end_idx] < 0)

        if (is_positive or is_negative):
            end_idx += 1

            while (end_idx < len(flattened)) and ((is_positive and flattened[end_idx] >= 0) or (is_negative and flattened[end_idx] <= 0)):
                end_idx += 1

        groups.append(flattened[current_idx:end_idx])

        current_size = end_idx - current_idx
        current_idx += current_size
        current_size = min_group_size

    return groups


def combine_groups(groups: List[np.ndarray], num_features: int) -> np.ndarray:
    """
    Combines the given groups back into a 2d measurement matrix.

    Args:
        groups: A list of 1d, flattened groups
        num_features: The number of features in each measurement (D)
    Returns:
        A [K, D] array containing the recovered measurements.
    """
    flattened = np.concatenate(groups)  # [K * D]
    return flattened.reshape(num_features, -1).T


def integer_part(x: float) -> int:
    """
    Returns the integer part of the given number.
    """
    return int(math.modf(x)[1])


def fractional_part(x: float) -> float:
    """
    Returns the fractional part of the given number
    """
    return math.modf(x)[0]


def get_signs(array: List[int]) -> List[int]:
    """
    Returns a binary array of the signs of each value.
    """
    return [1 if a >= 0 else 0 for a in array]


def apply_signs(array: List[int], signs: List[int]) -> List[int]:
    """
    Applies the signs to the given (absolute value) array.
    """
    assert len(array) == len(signs), 'Misaligned inputs ({0} vs {1})'.format(len(array), len(signs))
    return [x * (2 * s - 1) for x, s in zip(array, signs)]


def fixed_point_integer_part(fixed_point_val: int, precision: int) -> int:
    """
    Extracts the integer part from the given fixed point value.
    """
    if (precision >= 0):
        return fixed_point_val >> precision
    
    return fixed_point_val << precision


def fixed_point_frac_part(fixed_point_val: int, precision: int) -> int:
    """
    Extracts the fractional part from the given fixed point value.
    """
    if (precision >= 0):
        mask = (1 << precision) - 1
        return fixed_point_val & mask
    
    return 0

def num_bits_for_value(x: int) -> int:
    """
    Calculates the number if bits required to
    represent the given integer.
    """
    num_bits = 0
    while (x != 0):
        x = x >> 1
        num_bits += 1

    return max(num_bits, 1)


def run_length_encode(values: List[int], signs: List[int]) -> str:
    if len(values) <= 0:
        return ''

    current = abs(values[0])
    current_count = 1
    current_sign = signs[0]

    encoded: List[int] = []
    compressed_signs: List[int] = []
    reps: List[int] = []

    for i in range(1, len(values)):
        val = abs(values[i])

        if (val != current) or (signs[i] != current_sign):
            encoded.append(current)
            reps.append(current_count)
            compressed_signs.append(current_sign)

            current = val
            current_count = 1
            current_sign = signs[i]
        else:
            current_count += 1

    # Always include the final element
    encoded.append(current)
    reps.append(current_count)
    compressed_signs.append(current_sign)

    # Calculate the maximum number of bits needed to encode the values and repetitions
    max_encoded = np.max(np.abs(encoded))
    max_reps = np.max(np.abs(reps))

    encoded_bits = num_bits_for_value(max_encoded)
    reps_bits = num_bits_for_value(max_reps)

    encoded_values = pack(encoded, width=encoded_bits)
    encoded_reps = pack(reps, width=reps_bits)
    encoded_signs = pack(compressed_signs, width=1)

    metadata = ((encoded_bits << 4) | (reps_bits & 0xF)) & 0xFF
    metadata = ((len(encoded) << 8) | metadata) & 0xFFFFFF

    metadata_bytes = metadata.to_bytes(3, 'little')

    return metadata_bytes + encoded_values + encoded_reps + encoded_signs

def run_length_decode(encoded: bytes) -> List[int]:
    """
    Decodes the given RLE values.
    """
    metadata = int.from_bytes(encoded[0:3], 'little')
    encoded = encoded[3:]

    num_values = (metadata >> 8) & 0xFFF
    value_bits = (metadata >> 4) & 0xF
    rep_bits = metadata & 0xF

    value_bytes = int(math.ceil((num_values * value_bits) / BITS_PER_BYTE))
    rep_bytes = int(math.ceil((num_values * rep_bits) / BITS_PER_BYTE))
    sign_bytes = int(math.ceil(num_values / BITS_PER_BYTE))

    decoded_values = unpack(encoded[0:value_bytes], width=value_bits, num_values=num_values)
    encoded = encoded[value_bytes:]

    decoded_reps = unpack(encoded[0:rep_bytes], width=rep_bits, num_values=num_values)
    encoded = encoded[rep_bytes:]

    decoded_signs = unpack(encoded[0:sign_bytes], width=1, num_values=num_values)

    values: List[int] = []
    signs: List[int] = []

    for i in range(num_values):
        for j in range(decoded_reps[i]):
            values.append(decoded_values[i])
            signs.append(decoded_signs[i])

    return values, signs
