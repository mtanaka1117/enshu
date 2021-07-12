from adaptiveleak.utils.constants import PERIOD, BT_FRAME_SIZE
from adaptiveleak.utils.encryption import AES_BLOCK_SIZE
from adaptiveleak.utils.data_utils import calculate_bytes, truncate_to_block
from adaptiveleak.utils.data_types import PolicyType, EncodingMode, CollectMode, EncryptionMode
from .energy_systems import EnergyUnit


MARGIN = 1e-2


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


def get_group_target_bytes(width: int, collection_rate: float, num_features: int, seq_length: int, encryption_mode: EncryptionMode, energy_unit: EnergyUnit, target_energy: float) -> int:
    """
    Calculates the number of bytes targeted by the group encoding policy.

    Args:
        width: The bit-width of each feature
        collection_rate: The collection 
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

    # Estimate the energy required to send the number of bytes
    rounded_bytes = truncate_to_block(standard_num_bytes, block_size=BT_FRAME_SIZE) - 1

    if encryption_mode == EncryptionMode.BLOCK:
        rounded_bytes = truncate_to_block(rounded_bytes, block_size=AES_BLOCK_SIZE)

    estimated_energy = energy_unit.get_energy(num_collected=num_collected,
                                              num_bytes=rounded_bytes,
                                              use_noise=False)

    # Adjust the number of sent bytes until we reach a lower energy level
    while (estimated_energy > target_energy) and (rounded_bytes >= BT_FRAME_SIZE):
        rounded_bytes -= BT_FRAME_SIZE

        if encryption_mode == EncryptionMode.BLOCK:
            rounded_bytes = truncate_to_block(rounded_bytes, block_size=AES_BLOCK_SIZE)

        estimated_energy = energy_unit.get_energy(num_collected=num_collected,
                                                  num_bytes=rounded_bytes,
                                                  use_noise=False)

    return rounded_bytes
