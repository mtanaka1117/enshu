import socket

import os.path
import numpy as np
import time
from argparse import ArgumentParser
from typing import Optional

from adaptiveleak.policies import make_policy, run_policy, Policy
from adaptiveleak.utils.constants import LENGTH_BYTES, LENGTH_ORDER
from adaptiveleak.utils.encryption import encrypt, EncryptionMode, add_hmac
from adaptiveleak.utils.loading import load_data
from adaptiveleak.utils.file_utils import read_json, read_pickle_gz, save_pickle_gz


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
        self._hmac_secret = bytes.fromhex('97de481ffae5701de4f927573772b667')

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
            num_samples = inputs.shape[0]
            limit = min(num_samples, max_sequences) if max_sequences is not None else num_samples

            seq_length = inputs.shape[1]

            for idx in range(limit):
                # Execute the policy on this sequence
                policy.reset()

                measurements, indices = run_policy(policy=policy,
                                                   sequence=inputs[idx])

                # Encode the measurements into one message
                message = policy.encode(measurements=measurements, collected_indices=indices)

                # Encrypt and send the message
                key = self._aes_key if encryption_mode == EncryptionMode.BLOCK else self._chacha_key
                encrypted_message = encrypt(message=message, key=key, mode=encryption_mode)

                # Pre-pend the message length to the front (4 bytes)
                length = len(encrypted_message).to_bytes(LENGTH_BYTES, byteorder=LENGTH_ORDER)

                encrypted_message = length + encrypted_message

                # Add the HMAC authentication
                tagged_message = add_hmac(encrypted_message, secret=self._hmac_secret)

                sock.sendall(tagged_message)

                # Wait for a very small amount of time to rate limit a bit.
                # The server take precautions to separate messages, but we 
                # play it safe and just add a small separation between sequences.
                time.sleep(1e-5)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--encryption', type=str, choices=['block', 'stream'], required=True)
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--max-num-samples', type=int)
    parser.add_argument('--should-compress', action='store_true')
    args = parser.parse_args()

    # Load the data
    inputs, _ = load_data(dataset_name=args.dataset, fold='all')

    # Read the parameters
    params = read_json(args.params)

    # Extract the encryption mode
    encryption_mode = EncryptionMode[args.encryption.upper()]

    # Make the policy
    policy = make_policy(name=params['name'],
                         target=params['target'],
                         num_features=inputs.shape[2],
                         seq_length=inputs.shape[1],
                         dataset=args.dataset,
                         encryption_mode=encryption_mode,
                         encoding=params.get('encoding', 'unknown'),
                         should_compress=args.should_compress)

    # Run the sensor
    sensor = Sensor(server_host='localhost', server_port=args.port)
    sensor.run(inputs=inputs,
               policy=policy,
               max_sequences=args.max_num_samples,
               encryption_mode=encryption_mode)

    print('Completed Sensor.')
