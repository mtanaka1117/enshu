import numpy as np
import math
from functools import reduce
from typing import List, Tuple

from adaptiveleak.utils.data_utils import array_to_fp, array_to_float, pack, unpack


def encode_collected_mask(collected_indices: List[int], seq_length: int) -> bytearray:
    """
    Creates a bit-mask denoting the sent measurements in the sequence.
    """
    mask = 0
    collected_idx = 0
    mask_idx = -1
    offset = 0

    num_bytes = int(math.ceil(seq_length / 8))
    masks: List[int] = [0 for _ in range(num_bytes)]

    for seq_idx in range(seq_length):
        if seq_idx % 8 == 0:
            mask_idx += 1
            offset = seq_idx

        if (collected_idx < len(collected_indices)) and (collected_indices[collected_idx] == seq_idx):
            masks[mask_idx] |= 1 << (seq_idx - offset)
            collected_idx += 1

    return bytearray(masks)


def decode_collected_mask(bitmask: bytes, seq_length: int) -> List[int]:
    """
    Decodes the collected bit-mask into a list of indices.
    """
    indices: List[int] = []

    mask_idx = -1
    offset = 0

    for seq_idx in range(seq_length):
        if seq_idx % 8 == 0:
            mask_idx += 1
            offset = seq_idx

        is_present = bitmask[mask_idx] & (1 << (seq_idx - offset))
        if is_present != 0:
            indices.append(seq_idx)

    return indices


def encode_byte_measurements(measurements: np.ndarray, collected_indices: List[int], seq_length: int, precision: int) -> bytes:
    """
    Encodes the measurements into single-byte features.

    Args:
        measurements: A [K, D] array of the raw (sub-sampled) measurements to transmit.
            K is the number of measurements and D is the feature size.
        collected_indices: A list of the indices for the K collected measurements
        seq_length: The length of the full sequence (T)
        precision: The fixed point precision
    Returns:
        A hex string that represents the encoded measurements.
    """
    assert len(measurements.shape) == 2, 'Must provide a 2d array of measurements.'

    # Encode the measurements
    encoded = array_to_fp(measurements, precision=precision, width=8)

    # Add the offset value (2^7 - 1) to ensure all positive values
    encoded = encoded + 128

    # Begin message with the precision
    message = [precision]

    # Flatten measurements array and convert to a proper list
    flattened = encoded.reshape(-1).astype(int).tolist()  # [K * D]
    message.extend(flattened)

    # Convert the collected indices into a byte array
    collected_mask = encode_collected_mask(collected_indices, seq_length=seq_length)

    # Convert to bytes
    return bytes(collected_mask + bytearray(message))


def decode_byte_measurements(byte_str: bytes, seq_length: int, num_features: int) -> Tuple[np.ndarray, List[int]]:
    """
    Decodes the given byte string into an array of measurements.

    Args:
        byte_str: The raw byte string containing the encoded measurements
        seq_length: The length of the full sequence (T)
        num_features: The number of features in each measurement (D)
    Returns:
        A [K, D] array of recovered measurements
    """
    # Retrieve the number of collected measurements
    num_mask_bytes = int(math.ceil(seq_length / 8))
    bitmask = byte_str[0:num_mask_bytes]

    collected_indices = decode_collected_mask(bitmask=bitmask,
                                              seq_length=seq_length)

    # Unpack the rest of the message
    int_array = [int(b) for b in bytearray(byte_str[num_mask_bytes:])]

    raw_values = np.array(int_array[1:])
    precision = int_array[0]

    # Subtract the offset value (2^7)
    raw_values = raw_values - 128

    # Decode the measurement values
    decoded = array_to_float(raw_values, precision=precision)

    # Reshape to the proper size
    return decoded.reshape(-1, num_features), collected_indices


def encode_grouped_measurements(measurements: np.ndarray, collected_indices: List[int], widths: List[int], seq_length: int, non_fractional: int) -> bytes:
    """
    Encodes the measurements into sets of grouped features with different widths.

    Args:
        measurements: A [K, D] array of the raw (sub-sampled) measurements to transmit.
            K is the number of measurements and D is the feature size.
        collected_indices: A list of the indices for the K collected measurements
        seq_length: The length of the full sequence (T)
        widths: A list of bit-widths for features in each group
        non_fractional: The number of non-fractional bits per value
    Returns:
        A hex string that represents the encoded measurements.
    """
    assert len(measurements.shape) == 2, 'Must provide a 2d array of measurements.'

    # Encode the group meta-data
    num_groups = len(widths)
    num_measurements = len(measurements)
    group_metadata = bytes([num_measurements, num_groups]) + bytes(widths)

    # Divide features into groups and encode separately
    flattened = measurements.reshape(-1)  # [K * D]

    group_size = int((measurements.shape[0] * measurements.shape[1]) / num_groups)

    encoded_groups: List[bytes] = []
    for group_idx, width in enumerate(widths):
        # Extract the features in this group
        start, end = group_idx * group_size, (group_idx + 1) * group_size
        group_features = flattened[start:end] 

        # Encode the features
        precision = width - non_fractional
        group_encoded = array_to_fp(group_features, precision=precision, width=width)

        # Add offset to ensure positive values
        group_encoded = group_encoded + (1 << (precision - 1))

        # Pack the features into a single bit-string
        group_packed = pack(group_encoded, width=width)

        encoded_groups.append(group_packed)

    # Aggregate the encoded groups into a single bit-string
    encoded = reduce(lambda x, y: x + y, encoded_groups)

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
    num_measurements = int(encoded[0])
    num_groups = int(encoded[1])
    group_widths = [int(encoded[i+2]) for i in range(num_groups)]

    encoded = encoded[num_groups+2:]

    start = 0
    group_size = int((num_measurements * num_features) / num_groups)

    # Unpack the measurement values
    features: List[float] = []

    for group_idx, width in enumerate(group_widths):
        num_bytes = int(math.ceil(width * group_size / 8))

        group_encoded = encoded[start:start+num_bytes] 
        group_features = unpack(group_encoded, width=width, num_values=group_size)

        # Remove the offset value
        precision = width - non_fractional
        group_features = [x - (1 << (precision - 1)) for x in group_features]

        raw_features = array_to_float(group_features, precision=precision)
        
        features.extend(raw_features)
        start += num_bytes

    # Ensure didn't grab any extra values
    features = features[0:(num_measurements * num_features)]

    # Reshape into a 2d array
    measurements = np.array(features).reshape(-1, num_features)

    return measurements, collected_indices


#measurements = np.array([[0.25, 0.5], [0.75, 0.5]])
#collected_idx = [0, 4]
#seq_length = 8
#widths = [5, 4]
#non_fractional = 2
#
#encoded = encode_grouped_measurements(measurements=measurements,
#                                      collected_indices=collected_idx,
#                                      seq_length=seq_length,
#                                      widths=widths,
#                                      non_fractional=non_fractional)
#
#print(encoded)
#
#decoded, collected = decode_grouped_measurements(encoded=encoded,
#                                      seq_length=seq_length,
#                                      num_features=2,
#                                      non_fractional=non_fractional)
#
#
#print(decoded)
#print(collected)

