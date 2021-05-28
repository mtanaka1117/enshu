import numpy as np
import os.path
import h5py
import socket
from argparse import ArgumentParser
from collections import namedtuple, Counter
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from typing import Optional, List, Tuple

from adaptiveleak.policies import make_policy, Policy
from adaptiveleak.utils.constants import LENGTH_BYTES, LENGTH_ORDER
from adaptiveleak.utils.analysis import normalized_mae, normalized_rmse
from adaptiveleak.utils.encryption import decrypt, EncryptionMode, verify_hmac, SHA256_LEN
from adaptiveleak.utils.loading import load_data
from adaptiveleak.utils.file_utils import read_json, save_json_gz, read_pickle_gz


Message = namedtuple('Message', ['mac', 'length', 'data', 'full'])


def parse_message(message_buffer: bytes) -> Tuple[Message, int]:
    """
    Splits the message buffer into fields.

    Args:
        message_buffer: The current message buffer
    Returns:
        A tuple of two elements:
            (1) The parsed message
            (2) The number of consumed bytes. The buffer
                should be advanced by this amount.
    """
    length_start = SHA256_LEN
    data_start = SHA256_LEN + LENGTH_BYTES

    mac = message_buffer[:length_start]

    length = int.from_bytes(message_buffer[length_start:data_start], byteorder=LENGTH_ORDER)
    data = message_buffer[data_start:data_start + length]
    full = message_buffer[length_start:data_start + length]

    message = Message(mac=mac, length=length, data=data, full=full)
    return message, length + data_start


def reconstruct_sequence(measurements: np.ndarray, collected_indices: List[int], seq_length: int) -> np.ndarray:
    """
    Reconstructs a sequence using a linear interpolation.

    Args:
        measurements: A [K, D] array of sub-sampled features
        collected_indices: A list of [K] indices of the collected features
        seq_length: The length of the full sequence (T)
    Returns:
        A [T, D] array of reconstructed measurements.
    """
    feature_list: List[np.ndarray] = []
    seq_idx = list(range(seq_length))

    # Get the 'final' value using a linear interpolation on the last two values (if necessary)
    if collected_indices[-1] < (seq_length - 1):
        feature_diff = measurements[-1] - measurements[-2]
        time_diff = collected_indices[-1] - collected_indices[-2]

        slope = feature_diff / time_diff
        step = (seq_length - 1) - collected_indices[-2]

        right = slope * step + measurements[-2]
        right = np.expand_dims(right, axis=0)

        collected_indices.append(seq_length - 1)
        measurements = np.concatenate([measurements, right], axis=0)


    for feature_idx in range(measurements.shape[1]):
        collected_features = measurements[:, feature_idx]  # [K]
        reconstructed = np.interp(x=seq_idx,
                                  xp=collected_indices,
                                  fp=collected_features,
                                  left=collected_features[0],
                                  right=collected_features[-1])

        feature_list.append(np.expand_dims(reconstructed, axis=-1))
        
    return np.concatenate(feature_list, axis=-1)  # [T, D]


class Server:
    """
    This class mimics a server that infers 'missing' objects
    and performs inference
    """
    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port

        # These encryption keys are kept secret from the attacker program
        self._aes_key = bytes.fromhex('349fdc00b44d1aaacaa3a2670fd44244')
        self._chacha_key = bytes.fromhex('6166867d13e4d3c1686a57b21a453755d38a78943de17d76cb43a72bd5965b00')
        self._hmac_secret = bytes.fromhex('97de481ffae5701de4f927573772b667')

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def run(self, inputs: np.ndarray, labels: np.ndarray, policy: Policy, max_sequences: Optional[int], encryption_mode: EncryptionMode, should_print: bool, output_folder: str):
        """
        Opens the server for connections.
        """
        # Validate inputs
        assert len(labels.shape) == 1, 'Labels must be a 1d array'
        assert len(inputs.shape) == 3, 'Inputs must be a 3d array'
        assert inputs.shape[0] == labels.shape[0], 'Labels ({0}) and Inputs ({1}) do not align.'.format(labels.shape[0], inputs.shape[0])

        # Unpack the shape
        num_samples = inputs.shape[0]
        seq_length = inputs.shape[1]
        num_features = inputs.shape[2]

        # Initialize lists for logging
        num_bytes: List[int] = []
        num_measurements: List[int] = []

        maes: List[float] = []
        rmses: List[float] = []

        label_list: List[int] = []
        reconstructed_list: List[np.ndarray] = []
        width_counts: Counter = Counter()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock: 
            # Bind the sensor to the selected host and port
            sock.bind((self.host, self.port))

            if should_print:
                print('Started Server.')

            # Listen on the given port and accept any inbound connections
            sock.listen()
            conn, addr = sock.accept()

            if should_print:
                print('Accepted connection from {0}'.format(addr))

            with conn:
                # Set the maximum number of samples
                limit = min(max_sequences, num_samples) if max_sequences is not None else num_samples

                # Create a buffer for messages
                message_buffer = bytearray()

                # Iterate over all samples
                for idx in range(limit):
                    # Receive the given sequence (large-enough buffer)
                    recv = conn.recv(5000)
                    message_buffer.extend(recv)

                    # Parse the message
                    parsed, consumed_bytes = parse_message(message_buffer)

                    # Move the message buffer
                    message_buffer = message_buffer[consumed_bytes:]

                    # Verify the MAC
                    verification = verify_hmac(mac=parsed.mac,
                                               message=parsed.full,
                                               secret=self._hmac_secret)
                    if not verification:
                        print('Could not verify MAC for sample {0}. Quitting.'.format(idx))
                        break

                    # Decrypt the message
                    key = self._aes_key if encryption_mode == EncryptionMode.BLOCK else self._chacha_key
                    message = decrypt(ciphertext=parsed.data, key=key, mode=encryption_mode)

                    # Decode the measurements
                    measurements, collected_indices, widths = policy.decode(message=message)

                    # Reconstruct the sequence by inferring the missing elements, [T, D]
                    reconstructed = reconstruct_sequence(measurements=measurements,
                                                         collected_indices=collected_indices,
                                                         seq_length=seq_length)

                    # Compute the reconstruction error in the measurements
                    mae = mean_absolute_error(y_true=inputs[idx],
                                              y_pred=reconstructed)

                    rmse = mean_squared_error(y_true=inputs[idx],
                                              y_pred=reconstructed,
                                              squared=False)

                    # Log the results of this sequence
                    num_bytes.append(len(parsed.data))
                    num_measurements.append(len(measurements))
                    
                    maes.append(mae)
                    rmses.append(rmse)

                    label_list.append(int(labels[idx]))
                    reconstructed_list.append(np.expand_dims(reconstructed, axis=0))

                    for width in widths:
                        width_counts[width] += 1

                    if ((idx + 1) % 100) == 0:
                        print('Completed {0} sequences.'.format(idx + 1))

                # Compute the aggregate scores across across all samples
                true = inputs[0:limit]  # [N, T, D]
                true = true.reshape(-1, num_features)

                print(true.shape)

                reconstructed = np.vstack(reconstructed_list)  # [N, T, D]
                pred = reconstructed.reshape(-1, num_features)

                print(pred.shape)

                mae = mean_absolute_error(y_true=true, y_pred=pred)
                norm_mae = normalized_mae(y_true=true, y_pred=pred)

                rmse = mean_squared_error(y_true=true, y_pred=pred, squared=False)
                norm_rmse = normalized_rmse(y_true=true, y_pred=pred)

                r2 = r2_score(y_true=true, y_pred=pred, multioutput='variance_weighted')

        # Save the results
        result_dict = {
            'mae': mae,
            'rmse': rmse,
            'norm_mae': norm_mae,
            'norm_rmse': norm_rmse,
            'r2_score': r2,
            'avg_bytes': np.average(num_bytes),
            'avg_measurements': np.average(num_measurements),
            'count': len(maes),
            'widths': width_counts,
            'all_mae': maes,
            'all_rmse': rmses,
            'num_bytes': num_bytes,
            'num_measurements': num_measurements,
            'labels': label_list,
            'encryption_mode': encryption_mode.name,
            'policy': policy.as_dict()
        }

        output_path = os.path.join(output_folder, '{0}_{1}.json.gz'.format(str(policy), int(policy.target * 100)))
        save_json_gz(result_dict, output_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--target', type=float, required=True)
    parser.add_argument('--encryption', type=str, choices=['block', 'stream'], required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--encoding', type=str, choices=['standard', 'group'], required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--max-num-samples', type=int)
    parser.add_argument('--should-compress', action='store_true')
    args = parser.parse_args()

    # Load the test data
    inputs, labels = load_data(dataset_name=args.dataset, fold='test')

    # Set the encryption mode
    encryption_mode = EncryptionMode[args.encryption.upper()]

    # Make the server
    server = Server(host='localhost', port=args.port)

    # Extract the parameters
    #params = read_json(args.params)

    # Make the policy
    policy = make_policy(name=args.policy,
                         target=args.target,
                         num_features=inputs.shape[2],
                         seq_length=inputs.shape[1],
                         dataset=args.dataset,
                         encryption_mode=encryption_mode,
                         encoding=args.encoding,
                         should_compress=args.should_compress)

    # Run the experiment
    server.run(inputs=inputs,
               labels=labels,
               max_sequences=args.max_num_samples,
               should_print=True,
               policy=policy,
               encryption_mode=encryption_mode,
               output_folder=args.output_folder)
