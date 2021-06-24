from adaptiveleak.utils.constants import PERIOD, LENGTH_BYTES
from adaptiveleak.utils.data_utils import calculate_bytes
from adaptiveleak.utils.types import PolicyType, EncodingMode, CollectMode, EncryptionMode
from .energy_systems import EnergyUnit


def convert_rate_to_energy(collection_rate: float, width: int, encryption_mode: EncryptionMode, seq_length: int, num_features: int) -> float:
    """
    Converts a target collection rate to an energy rate / sequences. This
    energy rate corresponds to power when considering the fixed time per sequence period.

    Args:
        collection_rate: The fraction of sequence elements to collect
        width: The bit width of each feature
        encryption_mode: The type of encryption algorithm (block or stream)
        seq_length: The number of elements per sequence
        num_features: The number of features in each sequence element
    """
    # Make the energy unit based on uniform sampling
    energy_unit = EnergyUnit(policy_type=PolicyType.UNIFORM,
                             encoding_mode=EncodingMode.STANDARD,
                             collect_mode=CollectMode.LOW,
                             seq_length=seq_length,
                             period=PERIOD)

    # Calculate the energy required to collect the target rate
    target_collected = int(collection_rate * seq_length)

    sent_bytes = calculate_bytes(width=width,
                                 num_collected=target_collected,
                                 num_features=num_features,
                                 seq_length=seq_length,
                                 encryption_mode=encryption_mode)

    sent_bytes += LENGTH_BYTES  # Account for the length field

    energy_per_seq = energy_unit.get_energy(num_collected=seq_length,
                                            num_bytes=sent_bytes,
                                            use_noise=False)

    return energy_per_seq
