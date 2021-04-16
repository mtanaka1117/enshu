import numpy as np
import math
import time
from functools import reduce
from typing import List, Tuple

from adaptiveleak.utils.constants import SHIFT_BITS, BITS_PER_BYTE
from adaptiveleak.utils.data_utils import array_to_fp, array_to_float, pack, unpack, select_range_shift


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


def encode_standard_measurements(measurements: np.ndarray, collected_indices: List[int], seq_length: int, width: int, precision: int) -> bytes:
    """
    Encodes the measurements into single-byte features.

    Args:
        measurements: A [K, D] array of the raw (sub-sampled) measurements to transmit.
            K is the number of measurements and D is the feature size.
        collected_indices: A list of the indices for the K collected measurements
        seq_length: The length of the full sequence (T)
        width: The bit-width of each feature
        precision: The fixed point precision of each feature
    Returns:
        A hex string that represents the encoded measurements.
    """
    assert len(measurements.shape) == 2, 'Must provide a 2d array of measurements.'

    # Encode the measurements
    encoded = array_to_fp(measurements, precision=precision, width=width)

    # Add the offset value to ensure all positive values
    encoded = encoded + (1 << (width - 1))

    # Flatten measurements array and pack into a single bit-string
    flattened = encoded.reshape(-1).astype(int).tolist()  # [K * D]
    encoded_measurements = pack(flattened, width=width)  # [K * D]

    # Convert the collected indices into a byte array
    collected_mask = encode_collected_mask(collected_indices, seq_length=seq_length)

    # Append fields together
    return collected_mask + encoded_measurements


def decode_standard_measurements(byte_str: bytes, seq_length: int, num_features: int, width: int, precision: int) -> Tuple[np.ndarray, List[int]]:
    """
    Decodes the given byte string into an array of measurements.

    Args:
        byte_str: The raw byte string containing the encoded measurements
        seq_length: The length of the full sequence (T)
        num_features: The number of features in each measurement (D)
        width: The bit-width of each feature
        precision: The fixed-point precision of each feature
    Returns:
        A [K, D] array of recovered measurements
    """
    # Retrieve the number of collected measurements
    num_mask_bytes = int(math.ceil(seq_length / BITS_PER_BYTE))
    bitmask = byte_str[0:num_mask_bytes]

    collected_indices = decode_collected_mask(bitmask=bitmask,
                                              seq_length=seq_length)

    # Unpack the rest of the message
    int_array = [int(b) for b in bytearray(byte_str[num_mask_bytes:])]

    decoded_values = unpack(encoded=byte_str[num_mask_bytes:],
                            width=width,
                            num_values=len(collected_indices) * num_features)

    # Subtract the offset value (2^{w-1})
    offset = 1 << (width - 1)
    raw_values = np.array([x - offset for x in decoded_values])

    # Decode the measurement values
    decoded = array_to_float(raw_values, precision=precision)

    # Reshape to the proper size
    return decoded.reshape(-1, num_features), collected_indices


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
    shifts: List[int] = []

    # Divide features into groups and encode separately
    flattened = measurements.reshape(-1)  # [K * D]

    encoded_groups: List[bytes] = []
    for group_idx, width in enumerate(widths):
        # Extract the features in this group
        start = group_idx * group_size
        end = (group_idx + 1) * group_size

        if group_idx == (num_groups - 1):
            group_features = flattened[start:]
        else:
            group_features = flattened[start:end]

        # Encode the features using a dynamic number of fractional bits
        precision = width - non_fractional

        t0 = time.time()

        shift = select_range_shift(measurements=group_features,
                                   width=width,
                                   precision=precision,
                                   num_range_bits=SHIFT_BITS)

        t1 = time.time()
        # print('Time to select range shift: {0}'.format(t1 - t0))

        group_encoded = array_to_fp(group_features,
                                    precision=precision + shift,
                                    width=width)

        t2 = time.time()
        # print('Time to convert to fixed point: {0}'.format(t2 - t1))

        # Add offset to ensure positive values
        offset = 1 << (width - 1)
        group_encoded = group_encoded + offset

        # Pack the features into a single bit-string
        group_packed = pack(group_encoded, width=width)

        t3 = time.time()
        # print('Time to pack measurements: {0}'.format(t3 - t2))

        encoded_groups.append(group_packed)
        shifts.append(shift)

    # Aggregate the encoded groups into a single bit-string
    encoded = reduce(lambda x, y: x + y, encoded_groups)

    # Encode the group meta-data
    group_widths = encode_group_widths(widths=widths, shifts=shifts)
    group_metadata = bytes([group_size, num_groups]) + group_widths

    # Convert the collected indices into a byte array
    collected_mask = encode_collected_mask(collected_indices, seq_length=seq_length)

    # Convert to bytes
    return bytes(collected_mask) + group_metadata + encoded


def decode_grouped_measurements(encoded: bytes, seq_length: int, num_features: int, non_fractional: int) -> Tuple[np.ndarray, List[int]]:
    """
    Decodes the encoded group of measurements into the raw measurement values and collected
    sequence indices.

    Args:
        encoded: The encoded message containing the grouped measurements
        seq_length: The true sequence length
        num_features: The number of features per measurement
        non_fractional: The number of non-fractional bits
    Returns:
        A tuple of two values:
            (1) A [K, D] array of measurement values
            (2) A list of the indices of the collected measurements
    """
    # Retrieve the number of collected measurements
    num_mask_bytes = int(math.ceil(seq_length / 8))
    bitmask = encoded[0:num_mask_bytes]

    collected_indices = decode_collected_mask(bitmask=bitmask,
                                              seq_length=seq_length)

    encoded = encoded[num_mask_bytes:]

    # Unpack the meta-data
    num_measurements = len(collected_indices)
    group_size = int(encoded[0])
    num_groups = int(encoded[1])
    total_elements = num_features * num_measurements

    encoded_widths = encoded[2:num_groups+2]
    widths, shifts = decode_group_widths(encoded=encoded_widths)

    # Unpack the measurements
    encoded = encoded[num_groups+2:]

    start = 0
    count = 0  # the number of features so far

    features: List[float] = []

    for group_idx, (width, shift) in enumerate(zip(widths, shifts)):
        # Get the number of elements in the current group
        num_elements = (total_elements - count) if (num_groups - 1) == group_idx else group_size

        num_bytes = int(math.ceil((width * num_elements) / 8))

        group_encoded = encoded[start:start+num_bytes]
        group_features = unpack(group_encoded, width=width, num_values=num_elements)

        # Remove the offset value
        precision = width - non_fractional
        group_features = [x - (1 << (width - 1)) for x in group_features]

        raw_features = array_to_float(group_features, precision=precision + shift)
        
        features.extend(raw_features)
        start += num_bytes
        count += num_elements

    # Ensure didn't grab any extra values
    features = features[0:(num_measurements * num_features)]

    # Reshape into a 2d array
    measurements = np.array(features).reshape(-1, num_features)

    return measurements, collected_indices


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


def delta_encode(measurements: np.ndarray) -> np.ndarray:
    """
    Encodes the given measurements using the difference
    between consecutive features.

    Args:
        measurements: A [K, D] array of collected features
    Returns:
        A [K, D] array of delta encoded measurements. The first
        value is a true feature vector, and the remaining values
        are the consecutive differences.
    """
    assert len(measurements.shape) == 2, 'Must provide a 2d array'

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
    assert len(measurements.shape) == 2, 'Must provide a 2d array'

    initial = np.expand_dims(measurements[0], axis=0)  # [1, D]
    deltas = np.cumsum(measurements[1:], axis=0)  # [K - 1, D]

    return np.vstack([initial, initial + deltas])
