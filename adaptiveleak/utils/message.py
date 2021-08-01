import numpy as np
import math
import time
import bz2
from functools import reduce, partial
from typing import List, Tuple

from adaptiveleak.utils.constants import SHIFT_BITS, BITS_PER_BYTE, SMALL_NUMBER, MAX_SHIFT_GROUPS, MIN_WIDTH
from adaptiveleak.utils.data_utils import array_to_fp, array_to_float, pack, unpack, select_range_shift, to_fixed_point, to_float, get_signs, num_bits_for_value
from adaptiveleak.utils.data_utils import run_length_encode, run_length_decode, integer_part, fractional_part, apply_signs, select_range_shifts_array
from adaptiveleak.utils.data_utils import fixed_point_integer_part, fixed_point_frac_part, balance_group_size, array_to_fp_shifted, array_to_float_shifted, set_widths
from adaptiveleak.utils.shifting import merge_shift_groups


def encode_collected_mask(collected_indices: List[int], seq_length: int) -> bytes:
    """
    Creates a bit-mask denoting the sent measurements in the sequence.
    """
    mask = 0
    collected_idx = 0
    mask_idx = -1
    offset = 0

    num_bytes = int(math.ceil(seq_length / BITS_PER_BYTE))
    masks: List[int] = [0 for _ in range(num_bytes)]

    for seq_idx in range(seq_length):
        if (seq_idx % BITS_PER_BYTE) == 0:
            mask_idx += 1
            offset = seq_idx

        if (collected_idx < len(collected_indices)) and (collected_indices[collected_idx] == seq_idx):
            masks[mask_idx] |= 1 << (seq_idx - offset)
            collected_idx += 1

    return bytes(masks)


def decode_collected_mask(bitmask: bytes, seq_length: int) -> List[int]:
    """
    Decodes the collected bit-mask into a list of indices.
    """
    indices: List[int] = []

    mask_idx = -1
    offset = 0

    for seq_idx in range(seq_length):
        if (seq_idx % BITS_PER_BYTE) == 0:
            mask_idx += 1
            offset = seq_idx

        is_present = bitmask[mask_idx] & (1 << (seq_idx - offset))
        if is_present != 0:
            indices.append(seq_idx)

    return indices


def encode_standard_measurements(measurements: np.ndarray, collected_indices: List[int], seq_length: int, width: int, precision: int, should_compress: bool) -> bytes:
    """
    Encodes the measurements into single-byte features.

    Args:
        measurements: A [K, D] array of the raw (sub-sampled) measurements to transmit.
            K is the number of measurements and D is the feature size.
        collected_indices: A list of the indices for the K collected measurements
        seq_length: The length of the full sequence (T)
        width: The bit-width of each feature
        precision: The fixed point precision of each feature
        should_compress: Whether the function should compress the measurements after encoding
    Returns:
        A hex string that represents the encoded measurements.
    """
    assert len(measurements.shape) == 2, 'Must provide a 2d array of measurements.'

    # Flatten the measurements into a 1d array
    flattened = measurements.T.reshape(-1)

    # Quantize the measurements
    quantized = array_to_fp(flattened, precision=precision, width=width)

    if should_compress:
        # Delta encode the features
        flattened = delta_encode(quantized)

        # Take the absolute value of the flattened values
        abs_values = np.abs(flattened)

        # Split the features into integer and fractional parts
        integer_parts = list(map(partial(fixed_point_integer_part, precision=precision), abs_values))

        frac_fn = np.vectorize(partial(fixed_point_frac_part, precision=precision))
        fractional_parts = frac_fn(abs_values)

        # Encode the integer part using RLE
        integer_values = np.abs(integer_parts)
        integer_signs = get_signs(flattened)

        encoded_integers = run_length_encode(integer_values, integer_signs)

        encoded_fracs = pack(fractional_parts.astype(int).tolist(), width=precision)

        int_part_length = len(encoded_integers).to_bytes(2, 'little')
        encoded_measurements = int_part_length + encoded_integers + encoded_fracs

        encoded_measurements = bz2.compress(encoded_measurements)
    else:
        # Add the offset value to ensure all positive values
        encoded = quantized + (1 << (width - 1))

        # Flatten measurements array and pack into a single bit-string
        encoded_list = encoded.astype(int).tolist()  # [K * D]

        encoded_measurements = pack(encoded_list, width=width)  # [K * D]

    # Convert the collected indices into a byte array
    collected_mask = encode_collected_mask(collected_indices, seq_length=seq_length)

    # Append fields together
    return collected_mask + encoded_measurements


def decode_standard_measurements(byte_str: bytes, seq_length: int, num_features: int, width: int, precision: int, should_compress: bool) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Decodes the given byte string into an array of measurements.

    Args:
        byte_str: The raw byte string containing the encoded measurements
        seq_length: The length of the full sequence (T)
        num_features: The number of features in each measurement (D)
        width: The bit-width of each feature
        precision: The fixed-point precision of each feature
        should_compress: Whether the measurements were compressed during encoding
    Returns:
        A [K, D] array of recovered measurements
    """
    # Retrieve the number of collected measurements
    num_mask_bytes = int(math.ceil(seq_length / BITS_PER_BYTE))
    bitmask = byte_str[0:num_mask_bytes]

    collected_indices = decode_collected_mask(bitmask=bitmask,
                                              seq_length=seq_length)

    # Unpack the rest of the message
    if should_compress:
        decompressed = bz2.decompress(byte_str[num_mask_bytes:])

        # Extract the length of the integer part
        int_part_length = int.from_bytes(decompressed[0:2], 'little')

        # Unpack the two numerical components
        encoded_integers = decompressed[2:2+int_part_length]
        encoded_fracs = decompressed[2+int_part_length:]

        decoded_integers, decoded_signs = run_length_decode(encoded_integers)

        decoded_fracs = unpack(encoded=encoded_fracs,
                               width=precision,
                               num_values=(len(collected_indices) * num_features))

        # Combine the numerical parts
        combined = [(v << precision) | frac for v, frac in zip(decoded_integers, decoded_fracs)]
        combined = apply_signs(combined, decoded_signs)

        # Decode the delta-encoded measurements
        decoded = delta_decode(np.array(combined))

        decoded = array_to_float(decoded, precision=precision)
    else:
        decoded_values = unpack(encoded=byte_str[num_mask_bytes:],
                                width=width,
                                num_values=len(collected_indices) * num_features)

        # Subtract the offset value (2^{w-1})
        offset = 1 << (width - 1)
        raw_values = np.array([x - offset for x in decoded_values])

        # Decode the measurement values
        decoded = array_to_float(raw_values, precision=precision)

    # Reshape to the proper size
    return decoded.reshape(num_features, -1).T, collected_indices, [width]


def encode_stable_measurements(measurements: np.ndarray,
                               collected_indices: List[int],
                               widths: List[int],
                               shifts: List[int],
                               group_sizes: List[int],
                               seq_length: int,
                               non_fractional: int) -> bytes:
    """
    Encodes the measurements into sets of grouped features with different widths.

    Args:
        measurements: A [K, D] array of the raw (sub-sampled) measurements to transmit.
            K is the number of measurements and D is the feature size.
        collected_indices: A list of the indices for the K collected measurements
        seq_length: The length of the full sequence (T)
        width: The bit width for each feature
        non_fractional: The number of non-fractional bits per value
        target_bytes: The target number of bytes for this message
    Returns:
        A hex string that represents the encoded measurements.
    """
    assert len(measurements.shape) == 2, 'Must provide a 2d array of measurements.'
    assert len(group_sizes) == len(widths), 'Must have an equal number of group sizes and widths'
    assert len(group_sizes) == len(shifts), 'Must have an equal number of group sizes and shifts' 

    # Collect the group meta-data
    num_measurements, num_features = measurements.shape

    # Divide features into groups and encode separately
    flattened = measurements.T.reshape(-1)  # [K * D]

    # Convert the collected indices into a byte array
    collected_mask = encode_collected_mask(collected_indices, seq_length=seq_length)

    # Ensure positive shift values for encoding
    shift_offset = (1 << (SHIFT_BITS - 1))
    shifts_to_encode = [s + shift_offset for s in shifts]

    # Encode the shifts
    encoded_shifts = encode_shifts(shifts=shifts_to_encode,
                                   reps=group_sizes,
                                   widths=widths,
                                   num_shift_bits=SHIFT_BITS,
                                   min_width=MIN_WIDTH)

    # Encode each group of features
    encoded_groups: List[bytes] = []
        
    start_idx = 0
    for size, shift, width in zip(group_sizes, shifts, widths):
        # Get the features for this group
        end_idx = start_idx + size
        group_features = flattened[start_idx:end_idx]

        # Encode the features
        precision = width - non_fractional
        quantized = array_to_fp(group_features,
                                width=width,
                                precision=precision - shift)

        quantized += (1 << (width - 1))

        encoded_group = pack(quantized, width=width)
        encoded_groups.append(encoded_group)

        start_idx += size

    # Pack the data features
    encoded_features = reduce(lambda x, y: x + y, encoded_groups)

    # Convert to bytes
    return collected_mask + encoded_shifts + encoded_features


def decode_stable_measurements(encoded: bytes, seq_length: int, num_features: int, non_fractional: int) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Decodes the given byte string into an array of measurements.

    Args:
        byte_str: The raw byte string containing the encoded measurements
        seq_length: The length of the full sequence (T)
        num_features: The number of features in each measurement (D)
        width: The bit-width of each feature
        precision: The fixed-point precision of each feature
        should_compress: Whether the measurements were compressed during encoding
    Returns:
        A [K, D] array of recovered measurements
    """
    # Retrieve the number of collected measurements
    num_mask_bytes = int(math.ceil(seq_length / BITS_PER_BYTE))
    bitmask = encoded[0:num_mask_bytes]

    collected_indices = decode_collected_mask(bitmask=bitmask,
                                              seq_length=seq_length)

    encoded = encoded[num_mask_bytes:]

    # Retrieve the shifts and run-length decode
    shifts, widths, reps, shift_bytes = decode_shifts(encoded=encoded, num_shift_bits=SHIFT_BITS, min_width=MIN_WIDTH)

    # Remove the shift offset
    shift_offset = (1 << (SHIFT_BITS - 1))
    shifts = [s - shift_offset for s in shifts]

    encoded = encoded[shift_bytes:]

    decoded_features: List[float] = []

    byte_idx = 0
    for size, shift, width in zip(reps, shifts, widths):
        # Get the encoded data for this group
        num_group_bytes = int(math.ceil((size * width) / BITS_PER_BYTE))
        encoded_group = encoded[byte_idx:byte_idx+num_group_bytes]

        # Unpack the group features
        raw_group_features = unpack(encoded=encoded_group,
                                    width=width,
                                    num_values=size)

        # Convert back to floating point
        offset = (1 << (width - 1))
        raw_group_features = [(x - offset) for x in raw_group_features]

        precision = width - non_fractional
        group_features = array_to_float(raw_group_features,
                                        precision=precision - shift)

        decoded_features.extend(group_features)
        byte_idx += num_group_bytes

    # Reshape the measurement values
    recovered_features = np.reshape(decoded_features, (num_features, -1)).T

    # Reshape to the proper size
    return recovered_features, collected_indices, widths


def encode_shifts(shifts: List[int], reps: List[int], widths: List[int], num_shift_bits: int, min_width: int) -> bytes:
    """
    Encodes the given run-length encoded shifts into a bit string.

    Args:
        shifts: A list of [K] precision shifts
        reps: A list of [K] repetitions
        widths: A list of [K] bit widths
        num_shift_bits: The number of bits per shift value
        min_width: The minimum bit width
    Returns:
        The shifts and reps encoded as a byte string
    """
    assert len(shifts) == len(reps), 'Must provide same number of reps ({0}) and shifts ({1})'.format(len(reps), len(shifts))
    assert num_shift_bits < BITS_PER_BYTE, 'Number of shift bits must be less than {0}'.format(BITS_PER_BYTE)
    
    # Get masks needed for shifts and bit-widths
    width_mask = (1 << (BITS_PER_BYTE - num_shift_bits)) - 1
    shift_mask = (1 << num_shift_bits) - 1

    # Encode the shift values
    group_data: List[int] = []
    for width, shift in zip(widths, shifts):
        packed_data = ((width - min_width) & width_mask) << num_shift_bits
        packed_data |= (shift & shift_mask)

        group_data.append(packed_data)

    encoded_groups = pack(group_data, width=BITS_PER_BYTE)

    # Encode the repetitions
    reps_width = num_bits_for_value(max(reps))
    encoded_reps = pack(reps, width=reps_width)

    # Encode the count and reps width
    num_shifts = len(shifts)
    combined = (num_shifts << 4) | (reps_width & 0xF)
    encoded_header = combined.to_bytes(1, 'little')

    # Compile the result
    return encoded_header + encoded_reps + encoded_groups


def decode_shifts(encoded: bytes, num_shift_bits: int, min_width: int) -> Tuple[List[int], List[int], List[int], int]:
    """
    Decodes the shifts into a list of shift values
    and repetitions.

    Args:
        encoded: The encoded shifts byte string (output of encode_shifts())
        num_shift_bits: The number of bits to use for shifts
        min_width: The minimum bit-width of each feature value
    Returns:
        A tuple with four elements.
            (1) The shift values
            (2) The width values
            (3) The repetitions
            (4) The number of consumed bytes
    """
    # Extract the header elements
    encoded_header = int(encoded[0])
    reps_width = encoded_header & 0xF
    num_shifts = ((encoded_header >> 4) & 0xF)

    encoded = encoded[1:]

    # Get the repetitions
    num_reps_bytes = int(math.ceil((num_shifts * reps_width) / BITS_PER_BYTE))
    encoded_reps = encoded[0:num_reps_bytes]
    reps = unpack(encoded_reps, width=reps_width, num_values=num_shifts)

    encoded = encoded[num_reps_bytes:]

    # Get the shifts and group bit widths
    shifts: List[int] = []
    widths: List[int] = []

    width_mask = (1 << (BITS_PER_BYTE - num_shift_bits)) - 1
    shift_mask = (1 << (num_shift_bits)) - 1

    for group_idx in range(num_shifts):
        packed_data = encoded[group_idx]
        shift = packed_data & shift_mask
        width = (packed_data >> num_shift_bits) & width_mask
    
        shifts.append(shift)
        widths.append(width + min_width)

    total_bytes = 1 + num_reps_bytes + num_shifts

    return shifts, widths, reps, total_bytes


def delta_encode(measurements: np.ndarray) -> np.ndarray:
    """
    Encodes the given measurements using the difference
    between consecutive features.

    Args:
        measurements: A [K, D] (or [D]) array of collected features
    Returns:
        A [K, D] (or [D]) array of delta encoded measurements. The first
        value is a true feature vector, and the remaining values
        are the consecutive differences.
    """
    assert len(measurements.shape) in (1, 2), 'Must provide a 1d or 2d array'

    if len(measurements.shape) == 1:
        initial = measurements[0]
        diffs = measurements[1:] - measurements[:-1]
        return np.concatenate([[initial], diffs])
    else:
        initial = np.expand_dims(measurements[0], axis=0)  # [1, D]
        diffs = measurements[1:] - measurements[:-1]  # [K - 1, D]
        return np.vstack([initial, diffs])


def delta_decode(measurements: np.ndarray) -> np.ndarray:
    """
    Decodes the given delta encoded measurements.

    Args:
        measurements: A [K, D] array of delta-encoded features
    Returns:
        The raw feature vectors
    """
    assert len(measurements.shape) in (1, 2), 'Must provide a 1d or 2d array'

    if len(measurements.shape) == 1:
        initial = measurements[0]
        deltas = np.cumsum(measurements[1:])
        return np.concatenate([[initial], initial + deltas])
    else:
        initial = np.expand_dims(measurements[0], axis=0)  # [1, D]
        deltas = np.cumsum(measurements[1:], axis=0)  # [K - 1, D]
        return np.vstack([initial, initial + deltas])
