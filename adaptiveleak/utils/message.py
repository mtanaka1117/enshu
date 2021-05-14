import numpy as np
import math
import time
import bz2
from functools import reduce, partial
from typing import List, Tuple

from adaptiveleak.utils.constants import SHIFT_BITS, BITS_PER_BYTE, BOUND_BITS, BOUND_ORDER, SMALL_NUMBER, MAX_SHIFT_GROUPS
from adaptiveleak.utils.data_utils import array_to_fp, array_to_float, pack, unpack, select_range_shift, to_fixed_point, to_float, get_signs, num_bits_for_value
from adaptiveleak.utils.data_utils import array_to_fp_unsigned, run_length_encode, run_length_decode, integer_part, fractional_part, apply_signs, select_range_shifts_array
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

        # Encode the fractional parts in fixed-point representation
        #fixed_point_fracs = array_to_fp_unsigned(fractional_parts,
        #                                         precision=precision,
        #                                         width=precision)

        # encoded_fracs = pack(fixed_point_fracs.astype(int).tolist(), width=precision)
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

        # Convert the fractional parts to floats
        # decoded_fracs = array_to_float(decoded_fracs, precision=precision)
        
        # Combine the numerical parts
        combined = [(v << precision) | frac for v, frac in zip(decoded_integers, decoded_fracs)]
        combined = apply_signs(combined, decoded_signs)

        # Decode the delta-encoded measurements
        decoded = delta_decode(np.array(combined))

        decoded = array_to_float(decoded, precision=precision)

        # decoded = np.array(combined)
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


def encode_grouped_measurements(measurements: np.ndarray, collected_indices: List[int], widths: List[int], seq_length: int, non_fractional: int, group_size: int) -> bytes:
    """
    Encodes the measurements into sets of grouped features with different widths.

    Args:
        measurements: A [K, D] array of the raw (sub-sampled) measurements to transmit.
            K is the number of measurements and D is the feature size.
        collected_indices: A list of the indices for the K collected measurements
        seq_length: The length of the full sequence (T)
        widths: A list of bit-widths for features in each group
        non_fractional: The number of non-fractional bits per value
        group_size: The number of features per group
    Returns:
        A hex string that represents the encoded measurements.
    """
    assert len(measurements.shape) == 2, 'Must provide a 2d array of measurements.'

    # Collect the group meta-data
    num_groups = len(widths)
    num_measurements, num_features = measurements.shape
    shifts: List[int] = []  # 3 bits allocated for precision shifts

    # Divide features into groups and encode separately
    flattened = measurements.T.reshape(-1)  # [K * D]

    encoded_groups: List[bytes] = []
    for group_idx, width in enumerate(widths):
        # Extract the features in this group
        start = group_idx * group_size
        end = (group_idx + 1) * group_size

        if group_idx == (num_groups - 1):
            group_features = flattened[start:]
        else:
            group_features = flattened[start:end]

        precision = width - non_fractional

        # Encode the features using a dynamic number of
        # fractional bits (starting at the range amount)
        shift = select_range_shift(measurements=group_features,
                                   width=width,
                                   precision=precision,
                                   num_range_bits=SHIFT_BITS,
                                   is_unsigned=False)

        group_encoded = array_to_fp(group_features,
                                    precision=precision - shift,
                                    width=width)

        # Add offset to ensure positive values
        offset = 1 << (width - 1)
        group_encoded += offset

        # Pack the features into a single bit-string
        group_packed = pack(group_encoded, width=width)

        encoded_groups.append(group_packed)
        shifts.append(shift)

    # Aggregate the encoded groups into a single bit-string
    encoded = reduce(lambda x, y: x + y, encoded_groups)

    # Encode the group meta-data
    group_widths = encode_group_widths(widths=widths, shifts=shifts)
    group_metadata = bytes([num_groups]) + group_widths

    # Convert the collected indices into a byte array
    collected_mask = encode_collected_mask(collected_indices, seq_length=seq_length)

    # Convert to bytes
    return bytes(collected_mask) + group_metadata + encoded


def decode_grouped_measurements(encoded: bytes, seq_length: int, num_features: int, non_fractional: int, max_group_size: int) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Decodes the encoded group of measurements into the raw measurement values and collected
    sequence indices.

    Args:
        encoded: The encoded message containing the grouped measurements
        seq_length: The true sequence length
        num_features: The number of features per measurement
        non_fractional: The number of non-fractional bits
        max_group_size: The maximum group size
    Returns:
        A tuple of two values:
            (1) A [K, D] array of measurement values
            (2) A list of the indices of the collected measurements
            (3) A list of the group widths
    """
    # Retrieve the number of collected measurements
    num_mask_bytes = int(math.ceil(seq_length / 8))
    bitmask = encoded[0:num_mask_bytes]

    collected_indices = decode_collected_mask(bitmask=bitmask,
                                              seq_length=seq_length)
    encoded = encoded[num_mask_bytes:]

    # Unpack the meta-data
    num_measurements = len(collected_indices)
    # group_size = int(encoded[0])
    num_groups = int(encoded[0])
    group_size = balance_group_size(max_group_size=max_group_size,
                                    num_collected=num_measurements,
                                    num_features=num_features)

    total_elements = num_features * num_measurements

    encoded_widths = encoded[1:num_groups+1]
    widths, shifts = decode_group_widths(encoded=encoded_widths)

    encoded = encoded[num_groups+1:]

    start = 0
    count = 0  # the number of features so far

    features: List[float] = []

    for group_idx, (width, shift) in enumerate(zip(widths, shifts)):
        # Get the number of elements in the current group
        num_elements = (total_elements - count) if (num_groups - 1) == group_idx else group_size

        num_bytes = int(math.ceil((width * num_elements) / BITS_PER_BYTE))

        group_encoded = encoded[start:start+num_bytes]
        group_features = unpack(group_encoded, width=width, num_values=num_elements)

        precision = width - non_fractional
        offset = 1 << (width - 1)

        # Remove the offset value
        group_features = [x - offset for x in group_features]
        raw_features = array_to_float(group_features, precision=precision - shift)

        features.extend(raw_features)
        start += num_bytes
        
        #count += (num_elements + 1)
        count += num_elements

    # Ensure didn't grab any extra values
    features = features[0:(num_measurements * num_features)]

    # Reshape into a 2d array
    measurements = np.array(features).reshape(num_features, num_measurements)

    return measurements.T, collected_indices, widths


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

    # Collect the group meta-data
    num_measurements, num_features = measurements.shape

    # Divide features into groups and encode separately
    flattened = measurements.T.reshape(-1)  # [K * D]

    # Convert the collected indices into a byte array
    collected_mask = encode_collected_mask(collected_indices, seq_length=seq_length)

    # Encode the bit width
    encoded_width = width.to_bytes(1, 'little')

    # Compute the shifts
    precision = width - non_fractional
    shifts = select_range_shifts_array(flattened,
                                       width=width,
                                       precision=precision,
                                       num_range_bits=SHIFT_BITS)

    # Merge the shifts and perform run-length encoding
    merged_shifts, merged_reps = merge_shift_groups(values=flattened,
                                                    shifts=shifts,
                                                    width=width,
                                                    precision=precision,
                                                    max_num_groups=MAX_SHIFT_GROUPS)

    # Expand the shifts according to the given group sizes
    recovered_shifts = np.repeat(merged_shifts, group_sizes)

    # Ensure positive shift values
    shift_offset = (1 << (SHIFT_BITS - 1))
    merged_shifts = [s + shift_offset for s in merged_shifts]

    # Encode the shifts
    encoded_shifts = encode_shifts(shifts=merged_shifts,
                                   reps=merged_reps,
                                   num_shift_bits=SHIFT_BITS)

    # Convert the measurement values to fixed point
    quantized = array_to_fp_shifted(flattened, width=width, precision=precision, shifts=recovered_shifts)

    quantized = quantized + (1 << (width - 1))

    # Pack the data features
    encoded_features = pack(quantized, width=width)

    # Convert to bytes
    return encoded_width + collected_mask + encoded_shifts + encoded_features


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
    # Retrieve the bit width
    width = int(encoded[0])

    encoded = encoded[1:]

    # Retrieve the number of collected measurements
    num_mask_bytes = int(math.ceil(seq_length / BITS_PER_BYTE))
    bitmask = encoded[0:num_mask_bytes]

    collected_indices = decode_collected_mask(bitmask=bitmask,
                                              seq_length=seq_length)

    encoded = encoded[num_mask_bytes:]

    # Retrieve the shifts and run-length decode
    shifts, reps, shift_bytes = decode_shifts(encoded=encoded, num_shift_bits=SHIFT_BITS)
    shifts = np.repeat(shifts, reps)

    # Remove the shift offset
    shift_offset = (1 << (SHIFT_BITS - 1))
    shifts -= shift_offset

    encoded = encoded[shift_bytes:]

    # Retrieve the data values
    decoded_values = unpack(encoded=encoded,
                            width=width,
                            num_values=len(collected_indices) * num_features)

    # Subtract the offset value (2^{w-1})
    offset = 1 << (width - 1)
    raw_values = np.array([x - offset for x in decoded_values])

    # Decode the measurement values
    precision = width - non_fractional
    decoded_flattened = array_to_float_shifted(raw_values, precision=precision, shifts=shifts)
    decoded_features = decoded_flattened.reshape(num_features, -1).T

    # Reshape to the proper size
    return decoded_features, collected_indices, [width]


def encode_group_widths(widths: List[int], shifts: List[int]) -> bytes:
    """
    Encodes the group widths and shifts into a single binary string.

    Args:
        widths: A list of 5-bit width values
        shifts: A list of 3-bit range shifts
    Returns:
        A bit string containing the encoded information.
    """
    assert len(widths) == len(shifts), 'Must provide the same number of widths ({0}) and shifts ({1})'.format(len(widths), len(shifts))

    # Fixed offset and range values
    max_shift = (1 << (SHIFT_BITS - 1)) - 1
    min_shift = -1 * (1 << (SHIFT_BITS - 1))  # This could overflow, but SHIFT_BITS is generally small (3)
    
    width_bits = BITS_PER_BYTE - SHIFT_BITS
    max_width = (1 << width_bits) - 1
    min_width = 0

    width_mask = (1 << width_bits) - 1
    shift_mask = (1 << SHIFT_BITS) - 1

    encoded: List[int] = []
    for w, s in zip(widths, shifts):
        # Handle overflow and ensure the encoded
        # shift value is positive
        if (s > max_shift):
            s = max_shift
        elif (s < min_shift):
            s = min_shift

        s = (s - min_shift) & shift_mask

        # Handle overflow in the width value
        if (w > max_width):
            w = max_width
        elif (w < min_width):
            w = min_width

        w &= width_mask

        # Pack both values into a single byte
        merged = w | (s << width_bits)
        merged &= 0xFF

        encoded.append(merged)

    return bytes(encoded)


def decode_group_widths(encoded: bytes) -> Tuple[List[int], List[int]]:
    """
    Encodes the group widths and shifts into a single binary string.

    Args:
        encoded: The encoded group widths (output of encode_group_widths())
    Returns:
        A tuple with two elements:
            (1) A list of width values
            (2) A list of range shifts
    """
    widths: List[int] = []
    shifts: List[int] = []

    # Fixed offset and masks
    min_shift = -1 * (1 << (SHIFT_BITS - 1))

    width_bits = BITS_PER_BYTE - SHIFT_BITS
    width_mask = (1 << width_bits) - 1
    shift_mask = (1 << SHIFT_BITS) - 1

    for byte in encoded:
        # Unpack the width
        w = byte & width_mask
        widths.append(w)

        # Unpack the shift
        s = ((byte >> width_bits) & shift_mask) + min_shift
        shifts.append(s)

    return widths, shifts


def encode_shifts(shifts: List[int], reps: List[int], num_shift_bits: int) -> bytes:
    """
    Encodes the given run-length encoded shifts into a bit string.

    Args:
        shifts: A list of [K] precision shifts
        reps: A list of [K] repetitions
        num_shift_bits: The number of bits per shift value
    Returns:
        The shifts and reps encoded as a byte string
    """
    assert len(shifts) == len(reps), 'Must provide same number of reps ({0}) and shifts ({1})'.format(len(reps), len(shifts))
    
    # Encode the count
    num_shifts = len(shifts)
    encoded_count = num_shifts.to_bytes(1, 'little')

    # Encode the shift values
    encoded_shifts = pack(shifts, width=num_shift_bits)

    # Encode the repetitions
    reps_width = num_bits_for_value(max(reps))
    encoded_reps = pack(reps, width=reps_width)

    # Encode the count and reps width
    num_shifts = len(shifts)
    combined = (num_shifts << 4) | (reps_width & 0xF)
    encoded_header = combined.to_bytes(1, 'little')

    # Compile the result
    return encoded_header + encoded_reps + encoded_shifts


def decode_shifts(encoded: bytes, num_shift_bits: int) -> Tuple[List[int], List[int], int]:
    """
    Decodes the shifts into a list of shift values
    and repetitions.

    Args:
        encoded: The encoded shifts byte string (output of encode_shifts())
    Returns:
        A tuple with three elements.
            (1) The shift values
            (2) The repetitions
            (3) The number of consumed bytes
    """
    # Extract the header elements
    encoded_header = int(encoded[0])
    reps_width = encoded_header & 0xF
    num_shifts = (encoded_header >> 4) & 0xF

    encoded = encoded[1:]

    # Get the repetitions
    num_reps_bytes = int(math.ceil((num_shifts * reps_width) / BITS_PER_BYTE))
    encoded_reps = encoded[0:num_reps_bytes]
    reps = unpack(encoded_reps, width=reps_width, num_values=num_shifts)

    encoded = encoded[num_reps_bytes:]

    # Get the shifts
    num_shift_bytes = int(math.ceil((num_shifts * num_shift_bits) / BITS_PER_BYTE))
    encoded_shifts = encoded[0:num_shift_bytes]
    shifts = unpack(encoded_shifts, width=num_shift_bits, num_values=num_shifts)

    total_bytes = 1 + num_reps_bytes + num_shift_bytes

    return shifts, reps, total_bytes


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
