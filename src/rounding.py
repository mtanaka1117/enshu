import numpy as np
import math
from typing import Tuple

from utils.data_utils import get_num_groups, calculate_grouped_bytes
from policies import Policy
from compression import BlockWidth


def quantize(measurements: np.ndarray, num_transmitted: int, policy: Policy, should_pad: bool) -> Tuple[np.ndarray, int]:
    """
    Quantizes the measurements using the given policy and group structure.
    """
    assert measurements.shape[0] == num_transmitted, 'Expected {0} measurements. Got {1}'.format(num_transmitted, measurements.shape[0])

    width_policy = policy.width_policy
    group_size = width_policy.group_size

    num_groups = get_num_groups(group_size=group_size,
                                num_transmitted=num_transmitted)

    result: List[np.ndarray] = []
    total_bytes = 0

    # Get the widths for each group
    widths = width_policy.get_width(num_transmitted=num_transmitted, should_pad=should_pad)
    if not isinstance(widths, list):
        widths = [widths]

    assert len(widths) == num_groups, 'Expected {0} widths. Got {1}'.format(num_groups, len(widths))

    for group_idx, data_idx in enumerate(range(0, num_transmitted, group_size)):
        group = measurements[data_idx:data_idx+group_size]

        quantized, byte_count = policy.quantize_seq(measurements=group,
                                                    num_transmitted=group.shape[0],
                                                    width=widths[group_idx],
                                                    should_pad=should_pad)
        result.append(quantized)
        total_bytes += byte_count

    # For block width policies, the policy will undershoot the number of bytes
    # and give room for padding.
    if not should_pad and isinstance(width_policy, BlockWidth):
        target_bytes = int(width_policy.target_bytes) 
        total_bytes = max(total_bytes, target_bytes)

    return np.vstack(result), total_bytes
