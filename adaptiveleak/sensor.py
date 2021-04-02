import socket
import h5py
import os.path
import numpy as np
from argparse import ArgumentParser
from typing import Optional

from adaptiveleak.policies import make_policy, run_policy, Policy
from adaptiveleak.utils.encryption import encrypt, EncryptionMode
from adaptiveleak.utils.message import encode_byte_measurements
from adaptiveleak.utils.loading import load_data


class Sensor:
    """
    Simulates the behavior of a sensor.
    """
    def __init__(self, server_host: str, server_port: int):
        self._server_host = server_host
        self._server_port = server_port

        # These encryption keys are kept secret from the attacker program
        self._aes_key = bytes.fromhex('349fdc00b44d1aaacaa3a2670fd44244')
        self._chacha_key = bytes.fromhex('6166867d13e4d3c1686a57b21a453755d38a78943de17d76cb43a72bd5965b00')

    @property
    def host(self) -> str:
        return self._server_host

    @property
    def port(self) -> int:
        return self._server_port

    def run(self, inputs: np.ndarray, policy: Policy, encryption_mode: EncryptionMode, max_sequences: Optional[int]):
        """
        Execute the sensor on the given number of sequences.

        Args:
            inputs: A [N, T, D] array of features (D) for each sequence element (T)
                and sample (N)
            policy: The sampling policy
            encryption_mode: The type of encryption algorithm used for communication
            max_sequences: The number of sequences to execute. If None,
                then the sensor will execute on all sequences.
        """
        assert len(inputs.shape) == 3, 'Must provide a 3d input'

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to the server
            sock.connect((self.host, self.port))

            # Set the maximum number of sequences
            limit = inputs.shape[0]
            limit = min(limit, max_sequences) if max_sequences is not None else limit

            seq_length = inputs.shape[1]

            for idx in range(limit):
                # Execute the policy on this sequence
                policy.reset()

                measurements, indices = run_policy(policy=policy,
                                                   sequence=inputs[idx])

                policy.step(seq_idx=idx, count=len(indices))

                # Encode the measurements into one message
                message = policy.encode(measurements=measurements, collected_indices=indices)

                # Encrypt and send the message
                encrypted_message = encrypt(message=message, key=self._aes_key, mode=encryption_mode)

                sock.sendall(encrypted_message)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--encryption', type=str, choices=['block', 'stream'], required=True)
    args = parser.parse_args()

    # Load the data
    inputs, _ = load_data(dataset_name=args.dataset, fold='test')

    # Make the policy
    policy = make_policy(name='uniform',
                         target=0.6,
                         width=8,
                         precision=6,
                         num_features=inputs.shape[2],
                         seq_length=inputs.shape[1])

    # Extract the encryption mode
    encryption_mode = EncryptionMode[args.encryption.upper()]

    # Run the sensor
    sensor = Sensor(server_host='localhost', server_port=50000)
    sensor.run(inputs=inputs,
               policy=policy,
               max_sequences=2,
               encryption_mode=encryption_mode)
