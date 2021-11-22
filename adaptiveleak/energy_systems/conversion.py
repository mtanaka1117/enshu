import os.path
import math
import numpy as np
from typing import Tuple

from adaptiveleak.utils.constants import PERIOD, BT_FRAME_SIZE, BIG_NUMBER, LENGTH_SIZE
from adaptiveleak.utils.encryption import AES_BLOCK_SIZE, CHACHA_NONCE_LEN
from adaptiveleak.utils.data_utils import calculate_bytes, truncate_to_block
from adaptiveleak.utils.data_types import PolicyType, EncodingMode, CollectMode, EncryptionMode
from adaptiveleak.utils.file_utils import iterate_dir, read_json_gz
from .energy_systems import EnergyUnit


MARGIN = 1e-2

NUM_PADDING_FRAMES = 2
PADDING_FRAMES_FACTOR = 500


def convert_rate_to_energy(collection_rate: float, width: int, encryption_mode: EncryptionMode, collect_mode: CollectMode, seq_length: int, num_features: int) -> float:
    """
    Converts a target collection rate to an energy rate / sequences. This
    energy rate corresponds to power when considering the fixed time per sequence period.

    Args:
        collection_rate: The fraction of sequence elements to collect
        width: The bit width of each feature
        encryption_mode: The type of encryption algorithm (block or stream)
        collect_mode: The type of collection mode
        seq_length: The number of elements per sequence
        num_features: The number of features in each sequence element
    """
    # Make the energy unit based on uniform sampling
    energy_unit = EnergyUnit(policy_type=PolicyType.UNIFORM,
                             encoding_mode=EncodingMode.STANDARD,
                             collect_mode=collect_mode,
                             encryption_mode=encryption_mode,
                             seq_length=seq_length,
                             num_features=num_features,
                             period=PERIOD)

    # Calculate the energy required to collect the target rate
    target_collected = int(collection_rate * seq_length)

    sent_bytes = calculate_bytes(width=width,
                                 num_collected=target_collected,
                                 num_features=num_features,
                                 seq_length=seq_length,
                                 encryption_mode=encryption_mode)

    energy_per_seq = energy_unit.get_energy(num_collected=target_collected,
                                            num_bytes=sent_bytes,
                                            use_noise=False)

    return energy_per_seq + MARGIN


def get_group_target_bytes(width: int,
                           collection_rate: float,
                           num_features: int,
                           seq_length: int,
                           encryption_mode: EncryptionMode,
                           energy_unit: EnergyUnit,
                           target_energy: float) -> int:
    """
    Calculates the number of bytes targeted by the group encoding policy.

    Args:
        width: The bit-width of each feature
        collection_rate: The collection rate (fraction)
        num_features: The number of features per measurement
        seq_length: The length of a full sequence
        encryption_mode: The type of encryption (block or stream)
        energy_unit: The simulated energy unit
        target_energy: The targeted energy consumption per sequence
    Returns:
        The total number of bytes to target.
    """
    # Get the target number of collected elements
    num_collected = int(collection_rate * seq_length)

    # Calculate the number of bytes used by the standard policy
    standard_num_bytes = calculate_bytes(width=width,
                                         num_collected=num_collected,
                                         num_features=num_features,
                                         seq_length=seq_length,
                                         encryption_mode=encryption_mode)

    # Estimate the energy required to send the number of bytes. We start by (conservatively)
    # going multiple blocks under the given limit.
    rounded_bytes = standard_num_bytes
    num_padding_frames = NUM_PADDING_FRAMES + int(math.floor(rounded_bytes / PADDING_FRAMES_FACTOR))

    for _ in range(num_padding_frames):
        rounded_bytes = truncate_to_block(rounded_bytes, block_size=BT_FRAME_SIZE) - 1

    # Subtract out the meta-data bytes (be conservative here)
    metadata_bytes = LENGTH_SIZE + max(CHACHA_NONCE_LEN, AES_BLOCK_SIZE)

    data_bytes = rounded_bytes - metadata_bytes

    # Align with block encryption padding, as the block padding may put us into
    # the next communication frame. Stream ciphers wouldn't need this, but we
    # use this conservative approach to equalize the simulated communication
    # with the hardware results
    rounded_bytes = truncate_to_block(data_bytes, block_size=AES_BLOCK_SIZE) + metadata_bytes

    estimated_energy = energy_unit.get_energy(num_collected=num_collected,
                                              num_bytes=rounded_bytes,
                                              use_noise=False)

    # Adjust the number of sent bytes until we reach a lower energy level
    while (estimated_energy > target_energy) and (rounded_bytes >= BT_FRAME_SIZE):
        rounded_bytes = truncate_to_block(rounded_bytes, block_size=BT_FRAME_SIZE) - 1

        data_bytes = rounded_bytes - metadata_bytes
        rounded_bytes = truncate_to_block(data_bytes, block_size=AES_BLOCK_SIZE) + metadata_bytes

        estimated_energy = energy_unit.get_energy(num_collected=num_collected,
                                                  num_bytes=rounded_bytes,
                                                  use_noise=False)

    return rounded_bytes


def get_padded_collection_rate(dataset: str,
                               current_rate: float,
                               encryption_mode: str,
                               policy_type: str,
                               collect_mode: str,
                               width: int,
                               num_features: int,
                               seq_length: int) -> Tuple[float, int]:
    """
    Adjusts the collection rate for padded policies.

    Args:
        dataset: The name of the dataset
        current_rate: The existing collection rate
        encryption_mode: The name of the encryption type (block or stream)
        policy_type: The name of the policy
        collect_mode: The name of the collection mode (tiny, small, low, or high)
        width: The bit width of each feature
        num_features: The number of features per measurement
        seq_length: The number of measurements per sequence
    Returns:
        The adjusted collection rate
    """
    # Create the energy unit
    energy_unit = EnergyUnit(policy_type=PolicyType[policy_type.upper()],
                             encryption_mode=EncryptionMode[encryption_mode.upper()],
                             collect_mode=CollectMode[collect_mode.upper()],
                             encoding_mode=EncodingMode.PADDED,
                             seq_length=seq_length,
                             num_features=num_features,
                             period=PERIOD)

    # Get the target energy budget per sequence
    target_energy = convert_rate_to_energy(collection_rate=current_rate,
                                           width=width,
                                           encryption_mode=EncryptionMode[encryption_mode.upper()],
                                           collect_mode=CollectMode[collect_mode.upper()],
                                           num_features=num_features,
                                           seq_length=seq_length)

    # Get the directory of the existing logs
    base = os.path.dirname(__file__)

    policy_name = '{0}_standard'.format(policy_type.lower())
    log_folder = os.path.join(base, '..', 'saved_models', dataset, collect_mode.lower(), policy_name)

    best_diff = BIG_NUMBER
    best_rate = current_rate
    best_collected = 0

    min_rate = BIG_NUMBER
    min_collected = BIG_NUMBER

    did_find = False

    for log_path in iterate_dir(log_folder, pattern='.*json.gz'):
        standard_results = read_json_gz(log_path)

        # Get the max number of elements
        num_elements = max(standard_results['num_measurements'])
        num_bytes = calculate_bytes(width=width,
                                    num_collected=num_elements,
                                    num_features=num_features,
                                    seq_length=seq_length,
                                    encryption_mode=EncryptionMode[encryption_mode.upper()])

        avg_collected = int(math.ceil(np.average(standard_results['num_measurements'])))
        estimated_energy = energy_unit.get_energy(num_collected=avg_collected,
                                                  num_bytes=num_bytes,
                                                  use_noise=False)

        diff = abs(target_energy - estimated_energy)
        rate = standard_results['policy']['collection_rate']

        if rate < min_rate:
            min_rate = rate
            min_collected = num_elements

        if (diff < best_diff) and (target_energy >= estimated_energy):
            best_diff = diff
            best_rate = rate
            best_collected = num_elements

            did_find = True

    if did_find:
        return best_rate, best_collected

    return min_rate, min_collected
