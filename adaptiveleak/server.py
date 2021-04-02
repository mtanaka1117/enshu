import numpy as np
import os.path
import h5py
import socket
from argparse import ArgumentParser
from sklearn.metrics import mean_squared_error
from typing import Optional, List

from adaptiveleak.policies import make_policy, Policy
from adaptiveleak.utils.encryption import decrypt, EncryptionMode
from adaptiveleak.utils.message import decode_byte_measurements
from adaptiveleak.utils.loading import load_data


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

    estimate = np.zeros(shape=(measurements.shape[-1], 1))  # [D, 1]

    reconstructed: List[np.ndarray] = []
    for seq_idx in range(seq_length):
        if (collected_idx < len(collected_indices)) and (seq_idx == collected_indices[collected_idx]):
            estimate = measurements[collected_idx].reshape(-1, 1)  # [D, 1]
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

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def run(self, inputs: np.ndarray, labels: np.ndarray, policy: Policy, max_sequences: Optional[int], encryption_mode: EncryptionMode, should_print: bool):
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
                print('Started Server. Waiting for a connection...')

            # Listen on the given port and accept any inbound connections
            sock.listen()
            conn, addr = sock.accept()

            if should_print:
                print('Accepted connection from {0}'.format(addr))

            with conn:
                # Set the maximum number of samples
                limit = min(max_sequences, num_samples) if max_sequences is not None else num_samples

                # Iterate over all samples
                for idx in range(limit):
                    # Receive the given sequence (large-enough buffer)
                    recv = conn.recv(1024)

                    # Decrypt the message
                    message = decrypt(ciphertext=recv, key=self._aes_key, mode=encryption_mode)

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
                    num_bytes.append(len(recv))
                    num_measurements.append(len(measurements))
                    errors.append(error)
                    label_list.append(int(labels[idx]))

        # Save the results
        result_dict = {
            'errors': errors,
            'num_bytes': num_bytes,
            'num_measurements': num_measurements,
            'labels': label_list,
            'encryption_mode': encryption_mode.name,
            'policy': policy.as_dict(),
        }

        print(result_dict)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--encryption', type=str, choices=['block', 'stream'], required=True)
    args = parser.parse_args()

    # Load the test data
    inputs, labels = load_data(dataset_name=args.dataset, fold='test')

    # Set the encryption mode
    encryption_mode = EncryptionMode[args.encryption.upper()]

    # Make the server
    server = Server(host='localhost', port=50000)

    # Make the policy
    policy = make_policy(name='uniform',
                         target=0.6,
                         width=8,
                         precision=6,
                         num_features=inputs.shape[2],
                         seq_length=inputs.shape[1])

    # Run the experiment
    server.run(inputs=inputs,
               labels=labels,
               max_sequences=2,
               should_print=True,
               policy=policy,
               encryption_mode=encryption_mode)
