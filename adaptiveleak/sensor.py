import socket

import os.path
import numpy as np
import time
from argparse import ArgumentParser
from typing import Optional

from adaptiveleak.policies import BudgetWrappedPolicy, run_policy
from adaptiveleak.utils.constants import LENGTH_SIZE, LENGTH_ORDER, ENCODING, ENCRYPTION, COLLECTION, POLICIES
from adaptiveleak.utils.data_utils import array_to_fp, array_to_float
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

    def run(self, inputs: np.ndarray, policy: BudgetWrappedPolicy, num_sequences: int):
        """
        Execute the sensor on the given number of sequences.

        Args:
            inputs: A [N, T, D] array of features (D) for each sequence element (T)
                and sample (N)
            policy: The sampling policy
            encryption_mode: The type of encryption algorithm used for communication
            num_sequences: The number of sequences to execute.
        """
        assert len(inputs.shape) == 3, 'Must provide a 3d input'

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to the server
            sock.connect((self.host, self.port))

            # Get the sequence length from the input shape
            seq_length = inputs.shape[1]

            for idx in range(num_sequences):
                # Execute the policy on this sequence. We do not enforce the budget
                # on the sensor and instead track the energy on the server. We take this design
                # decision because the server logs all information.
                policy.reset()
                policy_result = run_policy(policy=policy,
                                           sequence=inputs[idx],
                                           should_enforce_budget=False)

                # Encode the measurements into one message
                message = policy.encode(measurements=policy_result.measurements,
                                        collected_indices=policy_result.collected_indices)

                # Encrypt and send the message
                key = self._aes_key if policy.encryption_mode == EncryptionMode.BLOCK else self._chacha_key
                encrypted_message = encrypt(message=message, key=key, mode=policy.encryption_mode)

                # Include the true number of collected measurements for proper energy logging
                true_num_collected = policy_result.num_collected.to_bytes(LENGTH_SIZE, byteorder=LENGTH_ORDER)

                # Include the message length to the front (2 bytes)
                length = len(encrypted_message).to_bytes(LENGTH_SIZE, byteorder=LENGTH_ORDER)

                # Concatenate all message fields
                encrypted_message = true_num_collected + length + encrypted_message

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
    parser.add_argument('--collection-rate', type=float, required=True)
    parser.add_argument('--encryption', type=str, choices=ENCRYPTION, required=True)
    parser.add_argument('--collect', type=str, choices=COLLECTION, required=True)
    parser.add_argument('--policy', type=str, choices=POLICIES, required=True)
    parser.add_argument('--encoding', type=str, choices=ENCODING, required=True)
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--max-num-seq', type=int)
    parser.add_argument('--should-compress', action='store_true')
    args = parser.parse_args()

    # Load the data
    inputs, _ = load_data(dataset_name=args.dataset, fold='test')

    # Unpack the input shape
    num_seq, seq_length, num_features = inputs.shape
    num_seq = min(num_seq, args.max_num_seq) if args.max_num_seq is not None else num_seq

    # Make the policy
    policy = BudgetWrappedPolicy(name=args.policy,
                                 collection_rate=round(args.collection_rate, 2),
                                 num_features=num_features,
                                 seq_length=seq_length,
                                 dataset=args.dataset,
                                 encryption_mode=args.encryption,
                                 collect_mode=args.collect,
                                 encoding=args.encoding,
                                 should_compress=args.should_compress)

    # Initialize the policy for the current budget
    policy.init_for_experiment(num_sequences=num_seq)

    # Pre-Quantize the data, as this is how the MCU reads the data
    quantized = array_to_fp(inputs, width=policy.width, precision=policy.precision)
    inputs = array_to_float(quantized, precision=policy.precision)

    # Run the sensor
    sensor = Sensor(server_host='localhost', server_port=args.port)
    sensor.run(inputs=inputs,
               policy=policy,
               num_sequences=num_seq)

    print('Completed Sensor.')
