import numpy as np
import time
import os
import sys
from argparse import ArgumentParser
from Cryptodome.Cipher import AES
from collections import namedtuple, Counter
from functools import reduce
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Optional, Iterable, List

from ble_manager import BLEManager
from adaptiveleak.server import reconstruct_sequence
from adaptiveleak.policies import EncodingMode
from adaptiveleak.utils.analysis import normalized_rmse, normalized_mae
from adaptiveleak.utils.constants import PERIOD, LENGTH_SIZE
from adaptiveleak.utils.encryption import AES_BLOCK_SIZE
from adaptiveleak.utils.data_utils import round_to_block
from adaptiveleak.utils.file_utils import save_json_gz, read_json, make_dir
from adaptiveleak.utils.loading import load_data
from adaptiveleak.utils.message import decode_standard_measurements, decode_stable_measurements


MAC_ADDRESS = '00:35:FF:13:A3:1E'
BLE_HANDLE = 18
HCI_DEVICE = 'hci0'

# Special bytes to handle sensor operation
RESET_BYTE = b'\xFF'
SEND_BYTE = b'\xBB'
START_BYTE = b'\xCC'

RESET_RESPONSE = b'\xCD'
START_RESPONSE = b'\xAB'

AES128_KEY = bytes.fromhex('349fdc00b44d1aaacaa3a2670fd44244')


def get_random_sequence(mean: List[float], std: List[float], seq_length: int, rand: np.random.RandomState) -> np.ndarray:
    rand_list: List[np.ndarray] = []

    for m, s in zip(mean, std):
        val = rand.normal(loc=m, scale=s, size=seq_length)  # [T]
        rand_list.append(np.expand_dims(val, axis=-1))

    return np.concatenate(rand_list, axis=-1)  # [T, D]


def execute_client(inputs: np.ndarray,
                   labels: np.ndarray,
                   output_file: str,
                   max_samples: Optional[int],
                   seq_length: int,
                   num_features: int,
                   width: int,
                   precision: int,
                   data_mean: np.ndarray,
                   data_std: np.ndarray,
                   encoding_mode: EncodingMode):
    """
    Starts the device client. This function either sends data and expects the device to respond with predictions
    or assumes that the device performs sensing on its own.

    Args:
        max_samples: Maximum number of sequences before terminating collection.
        data_files: Information about data files containing already-collected datasets.
        output_file: File path in which to save results
        start_idx: The starting index of the dataset
    """
    assert encoding_mode in (EncodingMode.STANDARD, EncodingMode.GROUP), 'Encoding type must be either standard or group'

    # Get the default number of non-fractional bits
    non_fractional = width - precision

    # Initialize the device manager
    device_manager = BLEManager(mac_addr=MAC_ADDRESS, handle=BLE_HANDLE, hci_device=HCI_DEVICE)

    # Lists to store experiment results
    num_bytes: List[int] = []
    num_measurements: List[int] = []
    maes: List[float] = []
    rmses: List[float] = []
    width_counter: Counter = Counter()

    reconstructed_list: List[np.ndarray] = []
    true_list: List[np.narray] = []
    label_list: List[int] = []

    count = 0
    recv_counter = 0

    rand = np.random.RandomState(seed=54121)

    print('==========')
    print('Starting experiment')
    print('==========')

    # Start the device
    try:
        device_manager.start()
        reset_result = device_manager.send_and_expect_byte(value=RESET_BYTE, expected=RESET_RESPONSE)

        if (reset_result == False):
            print('Could not reset the device.')
            return
    finally:
        device_manager.stop()

    print('Tested Connection. Press any key to start...')
    x = input()

    try:
        # Send the 'start' sequence
        device_manager.start()

        start_response = device_manager.send_and_expect_byte(value=START_BYTE, expected=START_RESPONSE)

        if (start_response == False):
            print('Could not start device.')
            return

        device_manager.stop()

        print('Sent Start signal.')
    
        for idx, (features, label) in enumerate(zip(inputs, labels)):
            if (max_samples is not None) and (idx >= max_samples):
                break

            # Delay to align with sampling period
            time.sleep(PERIOD)

            # Send the fetch character and wait for the response
            did_connect = device_manager.start(wait_time=0.1)

            if did_connect:
                response = device_manager.query(value=b'\xBB')
                device_manager.stop()

                message_byte_count = len(response)

                if (message_byte_count > 0):

                    # Extract the length and initialization vector
                    length = int.from_bytes(response[0:LENGTH_SIZE], 'big')
                    iv = response[LENGTH_SIZE:LENGTH_SIZE + AES_BLOCK_SIZE]

                    # Decrypt the response (Data already padded)
                    aes = AES.new(AES128_KEY, AES.MODE_CBC, iv)

                    data = response[LENGTH_SIZE + AES_BLOCK_SIZE:]

                    # Round up the length field
                    if length % AES_BLOCK_SIZE != 0:
                        rounded_length = round_to_block(length, block_size=AES_BLOCK_SIZE)
                    else:
                        rounded_length = length

                    data = data[:rounded_length]

                    if len(data) == rounded_length:
                        response = aes.decrypt(data)
                        response = response[0:length]

                        # Decode the response
                        if encoding_mode == EncodingMode.STANDARD:
                            measurements, collected_idx, widths = decode_standard_measurements(byte_str=response,
                                                                                               seq_length=seq_length,
                                                                                               num_features=num_features,
                                                                                               width=width,
                                                                                               precision=precision,
                                                                                               should_compress=False)
                            measurements = measurements.T.reshape(-1, num_features)
                        elif encoding_mode == EncodingMode.GROUP:
                            measurements, collected_idx, widths = decode_stable_measurements(encoded=response,
                                                                                             seq_length=seq_length,
                                                                                             num_features=num_features,
                                                                                             non_fractional=non_fractional)
                        else:
                            raise ValueError('Unknown encoding type: {0}'.format(encoding_mode))

                        # Interpolate the measurements
                        recovered = reconstruct_sequence(measurements=measurements,
                                                         collected_indices=collected_idx,
                                                         seq_length=seq_length)

                        # Log the widths
                        for w in widths:
                            width_counter[w] += 1

                        # Log the results of this sequence
                        num_bytes.append(message_byte_count)
                        num_measurements.append(len(measurements))

                        # Increment the received counter
                        recv_counter += 1
                    else:  # Count not decrypt the response -> Random Guessing
                        recovered = get_random_sequence(mean=data_mean,
                                                        std=data_std,
                                                        seq_length=seq_length,
                                                        rand=rand)
                else:   # Could not read response -> Random Guessing
                    recovered = get_random_sequence(mean=data_mean,
                                                    std=data_std,
                                                    seq_length=seq_length,
                                                    rand=rand)
            else: # Could not connect -> Random Guessing
                recovered = get_random_sequence(mean=data_mean,
                                                std=data_std,
                                                seq_length=seq_length,
                                                rand=rand)

            # Compute the error metrics
            mae = mean_absolute_error(y_true=features,
                                      y_pred=recovered)

            rmse = mean_squared_error(y_true=features,
                                     y_pred=recovered,
                                     squared=False)

            # Log the results of this sequence
            maes.append(float(mae))
            rmses.append(float(rmse))

            label_list.append(int(label))
            reconstructed_list.append(np.expand_dims(recovered, axis=0))
            true_list.append(np.expand_dims(features, axis=0))

            count += 1

            # Save the results so far
            results_dict = {
                'widths': width_counter,
                'recv_count': recv_counter,
                'count': count,
                'maes': maes,
                'rmses': rmses,
                'num_bytes': num_bytes,
                'num_measurements': num_measurements,
                'labels': label_list
            }
            save_json_gz(results_dict, output_file)

            print('Completed {0} Sequences. Avg MAE: {1:.4f}, Avg RMSE: {2:.4f}'.format(idx + 1, np.average(maes), np.average(rmses)), end='\r')

        print()
        reconstructed = np.vstack(reconstructed_list)  # [N, T, D]
        pred = reconstructed.reshape(-1, num_features)  # [N * T, D]

        true = np.vstack(true_list)  # [N, T, D]
        true = true.reshape(-1, num_features)  # [N * T, D]

        mae = mean_absolute_error(y_true=true, y_pred=pred)
        norm_mae = normalized_mae(y_true=true, y_pred=pred)

        rmse = mean_squared_error(y_true=true, y_pred=pred, squared=False)
        norm_rmse = normalized_rmse(y_true=true, y_pred=pred)

        r2 = r2_score(y_true=true, y_pred=pred, multioutput='variance_weighted')
    finally:
        device_manager.stop()

    print('Completed. MAE: {0:.4f}, RMSE: {1:.4f}'.format(mae, rmse))

    # Save the results
    results_dict = {
        'mae': float(mae),
        'norm_mae': float(norm_mae),
        'rmse': float(rmse),
        'norm_rmse': float(norm_rmse),
        'r2': float(r2),
        'avg_bytes': float(np.average(num_bytes)),
        'avg_measurements': float(np.average(num_measurements)),
        'widths': width_counter,
        'recv_count': recv_counter,
        'count': count,
        'maes': maes,
        'rmses': rmses,
        'num_bytes': num_bytes,
        'num_measurements': num_measurements,
        'labels': label_list
    }
    save_json_gz(results_dict, output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--collection-rate', type=float, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--encoding', type=str, required=True, choices=['standard', 'group'])
    parser.add_argument('--trial', type=int, required=True)
    parser.add_argument('--max-samples', type=int)
    args = parser.parse_args()

    make_dir(args.output_folder)
    output_file = os.path.join(args.output_folder, '{0}_{1}_{2}_trial{3}.json.gz'.format(args.policy, args.encoding, int(round(args.collection_rate * 100)), args.trial))

    if os.path.exists(output_file):
        print('The output file {0} exists. Do you want to overwrite it? [Y/N]'.format(output_file))
        d = input()
        if d.lower() not in ('y', 'yes'):
            sys.exit(0)

    # Read the data
    inputs, labels = load_data(dataset_name=args.dataset, fold='mcu')

    # Get the quantization parameters
    quantize_path = os.path.join('..', 'datasets', args.dataset, 'quantize.json')
    quantize = read_json(quantize_path)

    # Get the data distribution parameters
    distribution_path = os.path.join('..', 'datasets', args.dataset, 'distribution.json')
    distribution = read_json(distribution_path)

    execute_client(inputs=inputs,
                   labels=labels,
                   num_features=inputs.shape[2],
                   seq_length=inputs.shape[1],
                   max_samples=args.max_samples,
                   output_file=output_file,
                   width=quantize['width'],
                   precision=quantize['precision'],
                   data_mean=distribution['mean'],
                   data_std=distribution['std'],
                   encoding_mode=EncodingMode[args.encoding.upper()])
