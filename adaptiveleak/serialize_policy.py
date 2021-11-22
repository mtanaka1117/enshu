import h5py
import os.path
import numpy as np
import math
from argparse import ArgumentParser

from adaptiveleak.policies import make_policy, Policy, UniformPolicy, AdaptiveHeuristic, AdaptiveDeviation, AdaptivePolicy, SkipRNN
from adaptiveleak.utils.data_utils import to_fixed_point, array_to_fp, calculate_bytes, num_bits_for_value
from adaptiveleak.utils.constants import BITS_PER_BYTE, MAX_SHIFT_GROUPS, MIN_WIDTH, ENCODING, POLICIES, LENGTH_SIZE
from adaptiveleak.utils.encryption import AES_BLOCK_SIZE, CHACHA_NONCE_LEN
from adaptiveleak.utils.data_types import EncodingMode, EncryptionMode, CollectMode


OUTPUT_PATH = 'policy_parameters.h'
SKIP_RNN_PRECISION = 10
SKIP_RNN_WIDTH = 16


def serialize_matrix(mat: np.ndarray, name: str, width: int, precision: int, is_persistent: bool) -> str:
    """
    Converts the given matrix to fixed point and writes it to a C variable.
    """
    # Convert to fixed point
    fp = array_to_fp(mat.reshape(-1), width=width, precision=precision)

    num_rows, num_cols = mat.shape

    # Declare the data array
    data_str = ','.join(map(str, fp))
    data_var = 'static FixedPoint {0}_DATA[{1}] = {{ {2} }};\n'.format(name.upper(), num_rows * num_cols, data_str)

    # Declare the matrix
    mat_var = 'static struct Matrix {0} = {{ {0}_DATA, {1}, {2} }};\n'.format(name.upper(), num_rows, num_cols)

    if is_persistent:
        fram = '#pragma DATA_SECTION({0}_DATA, ".matrix")\n'.format(name.upper())
        return fram + data_var + mat_var

    return data_var + mat_var


def serialize_vector(vec: np.ndarray, name: str, width: int, precision: int) -> str:
    """
    Converts the given matrix to fixed point and writes it to a C variable.
    """
    # Flatten the vector
    vec = vec.reshape(-1)

    # Convert to fixed point
    fp = array_to_fp(vec, width=width, precision=precision)

    num_rows = vec.shape[0]

    # Declare the data array
    data_str = ','.join(map(str, fp))
    data_var = 'static FixedPoint {0}_DATA[{1}] = {{ {2} }};\n'.format(name.upper(), num_rows, data_str)

    # Declare the matrix
    mat_var = 'static struct Vector {0} = {{ {0}_DATA, {1} }};\n'.format(name.upper(), num_rows)

    return data_var + mat_var


def write_policy(policy: Policy, is_msp: bool):

    # Calculate the target number of bytes (same for all sequences)
    target_collected = int(policy.collection_rate * policy.seq_length)
    target_bytes = policy.target_bytes

    with open(OUTPUT_PATH, 'w') as fout:
        # Import necessary libraries
        fout.write('#include <stdint.h>\n')
        fout.write('#include "utils/matrix.h"\n\n')

        # Add the header guard
        fout.write('#ifndef POLICY_PARAMETERS_H_\n')
        fout.write('#define POLICY_PARAMETERS_H_\n')

        if is_msp:
            fout.write('#define IS_MSP\n')

        # Add information about the data
        fout.write('#define BITMASK_BYTES {0}\n'.format(int(math.ceil(policy.seq_length / 8))))
        fout.write('#define SEQ_LENGTH {0}\n'.format(policy.seq_length))
        fout.write('#define NUM_FEATURES {0}\n'.format(policy.num_features))
        fout.write('#define DEFAULT_WIDTH {0}\n'.format(policy.width))
        fout.write('#define DEFAULT_PRECISION {0}\n'.format(policy.precision))

        fout.write('#define TARGET_BYTES {0}\n'.format(target_bytes))
        fout.write('#define TARGET_DATA_BYTES {0}\n\n'.format(target_bytes - AES_BLOCK_SIZE - LENGTH_SIZE))

        if (not isinstance(policy, AdaptivePolicy)) or (policy.encoding_mode in (EncodingMode.STANDARD, EncodingMode.PADDED)):
            fout.write('#define IS_STANDARD_ENCODED\n')

            if policy.encoding_mode == EncodingMode.PADDED:
                fout.write('#define IS_PADDED\n')
        else:
            fout.write('#define IS_GROUP_ENCODED\n')

            size_width = num_bits_for_value(policy.seq_length)
            size_bytes = int(math.ceil((size_width * MAX_SHIFT_GROUPS) / BITS_PER_BYTE))
            mask_bytes = int(math.ceil(policy.seq_length / BITS_PER_BYTE))

            shift_bytes = 1 + MAX_SHIFT_GROUPS + size_bytes
            metadata_bytes = shift_bytes + mask_bytes

            if policy.encryption_mode == EncryptionMode.STREAM:
                metadata_bytes += CHACHA_NONCE_LEN
            else:
                metadata_bytes += AES_BLOCK_SIZE

            # Compute the target number of data bytes
            target_data_bytes = target_bytes - metadata_bytes
            target_data_bits = (target_data_bytes - MAX_SHIFT_GROUPS) * BITS_PER_BYTE

            # Estimate the maximum number of measurements we can collect
            max_features = int(target_data_bits / MIN_WIDTH)
            max_collected = int(max_features / policy.num_features)

            fout.write('#define MAX_COLLECTED {0}\n'.format(max_collected))

        # Add policy-specific information
        if isinstance(policy, UniformPolicy):
            fout.write('#define IS_UNIFORM\n')

            indices = policy._skip_indices
            fout.write('#define NUM_INDICES {0}\n'.format(len(indices)))

            serialized_indices = '{{{0}}}'.format(','.join(map(str, indices)))
            indices_variable = 'static const uint16_t COLLECT_INDICES[NUM_INDICES] = {0};'.format(serialized_indices)

            fout.write(indices_variable + '\n')
        elif isinstance(policy, AdaptiveHeuristic) or isinstance(policy, AdaptiveDeviation):

            if isinstance(policy, AdaptiveHeuristic):
                fout.write('#define IS_ADAPTIVE_HEURISTIC\n')
                precision = policy.precision
            else:
                fout.write('#define IS_ADAPTIVE_DEVIATION\n')

                precision = max(5, policy.precision)
                fout.write('#define DEVIATION_PRECISION {0}\n'.format(precision))

                alpha = to_fixed_point(policy._alpha, width=policy.width, precision=precision)
                beta = to_fixed_point(policy._beta, width=policy.width, precision=precision)

                fout.write('#define ALPHA {0}\n'.format(alpha))
                fout.write('#define BETA {0}\n'.format(beta))

            threshold = to_fixed_point(policy._threshold, width=policy.width, precision=precision)

            fout.write('#define THRESHOLD {0}\n'.format(threshold))
            fout.write('#define MAX_SKIP {0}\n'.format(policy.max_skip))
            fout.write('#define MIN_SKIP {0}\n'.format(policy.min_skip))
        elif isinstance(policy, SkipRNN):
            fout.write('#define IS_SKIP_RNN\n')

            # Unpack the relevant variables
            W_gates = policy._W_gates.T  # The MSP device will perform matmul on the transpose
            b_gates = policy._b_gates

            W_state = policy._W_state.T
            b_state = policy._b_state

            initial_state = policy._initial_state

            mean = policy._mean
            scale = policy._scale

            # Use the reciprocal scale to avoid division on the embedded device
            scale = 1.0 / scale

            # Split the gate variables into candidate and update components
            state_size = initial_state.shape[0]
            W_update = W_gates[:, :state_size]
            W_candidate = W_gates[:, state_size:]

            b_update = b_gates[:state_size]
            b_candidate = b_gates[state_size:]

            # Add 1 to the update bias to avoid doing this computation on the embedded device
            b_update += 1

            # Write the constant state size and RNN precision value
            fout.write('#define STATE_SIZE {0}\n'.format(state_size))
            fout.write('#define RNN_PRECISION {0}\n'.format(SKIP_RNN_PRECISION))

            # Convert the parameters to C variables
            fout.write(serialize_matrix(W_update, name='W_UPDATE', width=SKIP_RNN_WIDTH, precision=SKIP_RNN_PRECISION, is_persistent=is_msp))
            fout.write(serialize_matrix(W_candidate, name='W_CANDIDATE', width=SKIP_RNN_WIDTH, precision=SKIP_RNN_PRECISION, is_persistent=is_msp))

            fout.write(serialize_vector(b_update, name='B_UPDATE', width=SKIP_RNN_WIDTH, precision=SKIP_RNN_PRECISION))
            fout.write(serialize_vector(b_candidate, name='B_CANDIDATE', width=SKIP_RNN_WIDTH, precision=SKIP_RNN_PRECISION))

            fout.write(serialize_vector(W_state, name='W_STATE', width=SKIP_RNN_WIDTH, precision=SKIP_RNN_PRECISION))
            fout.write('static FixedPoint B_STATE = {0};\n'.format(to_fixed_point(b_state[0, 0], width=SKIP_RNN_WIDTH, precision=SKIP_RNN_PRECISION)))

            fout.write(serialize_vector(initial_state, name='INITIAL_STATE', width=SKIP_RNN_WIDTH, precision=SKIP_RNN_PRECISION))
            fout.write(serialize_vector(mean, name='MEAN', width=policy.width, precision=policy.precision))
            fout.write(serialize_vector(scale, name='SCALE', width=policy.width, precision=policy.precision))
        else:
            raise ValueError('Unknown policy: {0}'.format(policy))

        fout.write('#endif\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--policy', type=str, required=True, choices=POLICIES)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--collection-rate', type=float, required=True)
    parser.add_argument('--encoding', type=str, required=True, choices=ENCODING)
    parser.add_argument('--is-msp', action='store_true')
    args = parser.parse_args()

    # Load the data to get parameters of the task
    path = os.path.join('datasets', args.dataset, 'validation', 'data.h5')
    with h5py.File(path, 'r') as fin:
        inputs = fin['inputs'][:]

    seq_length = inputs.shape[1]
    num_features = inputs.shape[2]

    # Make the policy
    policy = make_policy(name=args.policy,
                         seq_length=seq_length,
                         num_features=num_features,
                         encryption_mode='block',
                         collect_mode='tiny',
                         encoding=args.encoding,
                         collection_rate=args.collection_rate,
                         dataset=args.dataset,
                         should_compress=False)

    write_policy(policy, is_msp=args.is_msp)
