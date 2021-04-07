import numpy as np
import os.path
import h5py
import socket
from argparse import ArgumentParser
from sklearn.metrics import mean_squared_error
from typing import Optional, List

from adaptiveleak.policies import make_policy, Policy
from adaptiveleak.utils.encryption import decrypt, EncryptionMode, verify_hmac, SHA256_LEN
from adaptiveleak.utils.message import decode_byte_measurements
from adaptiveleak.utils.loading import load_data
from adaptiveleak.utils.file_utils import read_json, save_json_gz, read_pickle_gz


def reconstruct_sequence(measurements: np.ndarray, collected_indices: List[int], seq_length: int) -> np.ndarray:
    """
    Reconstructs a sequence using a last-known policy.

    Args:
        measurements: A [K, D] array of sub-sampled features
        collected_indices: A list of [K] indices of the collected features
        seq_length: The length of the full sequence (T)
    Returns:
        A [T, D] array of reconstructed measurements.
    """
    collected_idx = 0

    num_features = measurements.shape[-1]
    estimate = np.zeros(shape=(1, num_features))  # [1, D]

    reconstructed: List[np.ndarray] = []
    for seq_idx in range(seq_length):
        if (collected_idx < len(collected_indices)) and (seq_idx == collected_indices[collected_idx]):
            estimate = measurements[collected_idx].reshape(1, num_features)  # [1, D]
            collected_idx += 1

        reconstructed.append(estimate)

    return np.vstack(reconstructed)


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
        errors: List[float] = []
        label_list: List[int] = []

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
                    recv = conn.recv(2048)
                    message_buffer.extend(recv)

                    # Extract the HMAC
                    mac = message_buffer[:SHA256_LEN]

                    # Get the information for the current sample
                    length = int.from_bytes(message_buffer[SHA256_LEN:SHA256_LEN+4], byteorder='little')
                    recv_sample = bytes(message_buffer[SHA256_LEN+4:SHA256_LEN+length+4])

                    # Move the message buffer
                    message_buffer = message_buffer[4+length+SHA256_LEN:]

                    # Verify the MAC
                    if not verify_hmac(mac=mac, message=recv_sample, secret=self._hmac_secret):
                        print('Could not verify MAC for sample {0}. Skipping.'.format(idx))
                        continue

                    # Decrypt the message
                    key = self._aes_key if encryption_mode == EncryptionMode.BLOCK else self._chacha_key
                    message = decrypt(ciphertext=recv_sample, key=key, mode=encryption_mode)

                    # Decode the measurements
                    measurements, collected_indices = policy.decode(message=message)

                    # Reconstruct the sequence by inferring the missing elements, [T, D]
                    reconstructed = reconstruct_sequence(measurements=measurements,
                                                         collected_indices=collected_indices,
                                                         seq_length=seq_length)

                    # Compute the reconstruction error in the measurements
                    error = mean_squared_error(y_true=inputs[idx],
                                               y_pred=reconstructed)

                    # Log the results of this sequence
                    num_bytes.append(len(recv_sample))
                    num_measurements.append(len(measurements))
                    errors.append(error)
                    label_list.append(int(labels[idx]))

                    if ((idx + 1) % 100) == 0:
                        print('Completed {0} sequences.'.format(idx + 1))

        # Save the results
        result_dict = {
            'avg_error': np.average(errors),
            'avg_bytes': np.average(num_bytes),
            'avg_measurements': np.average(num_measurements),
            'errors': errors,
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
    parser.add_argument('--encryption', type=str, choices=['block', 'stream'], required=True)
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--max-num-samples', type=int)
    args = parser.parse_args()

    # Load the test data
    inputs, labels = load_data(dataset_name=args.dataset, fold='test')

    # Set the encryption mode
    encryption_mode = EncryptionMode[args.encryption.upper()]

    # Make the server
    server = Server(host='localhost', port=args.port)

    # Extract the parameters
    params = read_json(args.params)

    # Get preset threshold (if present)
    threshold_path = os.path.join('saved_models', args.dataset, 'thresholds.pkl.gz')
    thresholds = read_pickle_gz(threshold_path) if os.path.exists(threshold_path) else dict()

    # Make the policy
    policy = make_policy(name=params['name'],
                         target=params['target'],
                         precision=params['precision'],
                         num_features=inputs.shape[2],
                         seq_length=inputs.shape[1],
                         encryption_mode=encryption_mode,
                         encoding=params.get('encoding', 'unknown'),
                         threshold=thresholds.get(params['target'], 0.0))

    # Run the experiment
    server.run(inputs=inputs,
               labels=labels,
               max_sequences=args.max_num_samples,
               should_print=True,
               policy=policy,
               encryption_mode=encryption_mode,
               output_folder=args.output_folder)
