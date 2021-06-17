import numpy as np
import time
import os
import sys
import h5py
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
from adaptiveleak.utils.file_utils import save_json_gz, read_json, make_dir
from adaptiveleak.utils.message import decode_standard_measurements, decode_stable_measurements


MAC_ADDRESS = '00:35:FF:13:A3:1E'
BLE_HANDLE = 18
HCI_DEVICE = 'hci1'
PERIOD = 7

AES128_KEY = bytes.fromhex('349fdc00b44d1aaacaa3a2670fd44244')


def execute_client(inputs: np.ndarray,
                   labels: np.ndarray,
                   output_file: str,
                   start_idx: int,
                   max_samples: Optional[int],
                   seq_length: int,
                   num_features: int,
                   width: int,
                   precision: int,
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

    print('==========')
    print('Starting experiment')
    print('==========')

    # Start and reset the device
    try:
        device_manager.start()

        time.sleep(0.1)
        device_manager.reset_device()
        time.sleep(0.1)
        #device_manager.send(value=b'\xFF\xCC')
    finally:
        device_manager.stop()

    start = time.time()

    print('Tested Connection. Press any key to start...')
    x = input()

    try:
        end = time.time()
        #time.sleep(1)

        # Send the 'start' sequence
        start = time.time()
        device_manager.start()

        time.sleep(0.1)
        device_manager.send(value=b'\xCC')
        time.sleep(0.1)

        device_manager.stop()
        end = time.time()

        print('Sent Start signal...')
    
        time.sleep(max(PERIOD - (end - start), 0))

        for idx, (features, label) in enumerate(zip(inputs, labels)):
            if idx < start_idx:
                continue

            if (max_samples is not None) and (idx >= max_samples):
                break

            # Send the fetch character and wait for the response
            start = time.time()

            device_manager.start()
            time.sleep(0.1)
            response = device_manager.query(value=b'\xBB')
            time.sleep(0.1)
            device_manager.stop()

            message_byte_count = len(response)

            # Extract the length
            length = int.from_bytes(response[0:2], 'big')

            # Decrypt the response
            aes = AES.new(AES128_KEY, AES.MODE_ECB)  # TODO: Get CBC Working
            response = aes.decrypt(response[2:])
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
                                                                                 non_fractional=width - precision)
            else:
                raise ValueError('Unknown encoding type: {0}'.format(encoding_mode))

            print(measurements)
            print(len(collected_idx))
            print(collected_idx)

            # Interpolate the measurements
            recovered = reconstruct_sequence(measurements=measurements,
                                             collected_indices=collected_idx,
                                             seq_length=seq_length)

            # Compute the error metrics
            mae = mean_absolute_error(y_true=features,
                                      y_pred=recovered)

            rmse = mean_squared_error(y_true=features,
                                     y_pred=recovered,
                                     squared=False)

            # Log the results of this sequence
            num_bytes.append(message_byte_count)
            num_measurements.append(len(measurements))
            
            maes.append(float(mae))
            rmses.append(float(rmse))

            label_list.append(int(label))
            reconstructed_list.append(np.expand_dims(recovered, axis=0))
            true_list.append(np.expand_dims(features, axis=0))

            for width in widths:
                width_counter[width] += 1

            end = time.time()

            time.sleep(max(PERIOD - (end - start), 1e-5))
            count += 1

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
        'count': count,
        'start_idx': start_idx,
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
    parser.add_argument('--target', type=float, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--encoding', type=str, required=True, choices=['standard', 'group'])
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-samples', type=int)
    args = parser.parse_args()

    make_dir(args.output_folder)
    output_file = os.path.join(args.output_folder, '{0}_{1}_{2}.json.gz'.format(args.policy, args.encoding, int(round(args.target * 100))))

    if os.path.exists(output_file):
        print('The output file {0} exists. Do you want to overwrite it? [Y/N]'.format(output_file))
        d = input()
        if d.lower() not in ('y', 'yes'):
            sys.exit(0)

    # Read the data
    with h5py.File(os.path.join('..', 'datasets', args.dataset, 'mcu', 'data.h5'), 'r') as fin:
        inputs = fin['inputs'][:]
        labels = fin['output'][:]

    # Get the quantization parameters
    quantize_path = os.path.join('..', 'datasets', args.dataset, 'quantize.json')
    quantize = read_json(quantize_path)

    execute_client(inputs=inputs,
                   labels=labels,
                   num_features=inputs.shape[2],
                   seq_length=inputs.shape[1],
                   max_samples=args.max_samples,
                   output_file=output_file,
                   start_idx=args.start_index,
                   width=quantize['width'],
                   precision=quantize['precision'],
                   encoding_mode=EncodingMode[args.encoding.upper()])
