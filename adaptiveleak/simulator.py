import os
import pexpect
import sys
import random
import numpy as np
import time
from argparse import ArgumentParser
from datetime import datetime
from typing import List

from adaptiveleak.utils.constants import POLICIES
from adaptiveleak.utils.file_utils import make_dir


MAX_RETRIES = 10
RETRY_SLEEP = 0.1
TIMEOUT = 10

SERVER_CMD_ALL = 'python server.py --dataset {0} --encryption {1} --policy {2} --encoding {3} --collection-rate {4} --output-folder {5} --port {6}'
SERVER_CMD_SAMPLES = 'python server.py --dataset {0} --encryption {1} --policy {2} --encoding {3} --collection-rate {4} --output-folder {5} --port {6} --max-num-seq {7}'

SENSOR_CMD_ALL = 'python sensor.py --dataset {0} --encryption {1} --policy {2} --encoding {3} --collection-rate {4} --port {5}'
SENSOR_CMD_SAMPLES = 'python sensor.py --dataset {0} --encryption {1} --policy {2} --encoding {3} --collection-rate {4} --port {5} --max-num-seq {6}'


def expect_with_retry(comm_module: pexpect.spawn, expected: str):
    has_recieved = False
    retry_counter = 0

    while (not has_recieved) and (retry_counter < MAX_RETRIES):
        try:
            server.expect(expected, timeout=TIMEOUT)
            has_recieved = True
        except pexpect.exceptions.TIMEOUT:
            retry_counter += 1
            time.sleep(RETRY_SLEEP)

    if (retry_counter >= MAX_RETRIES):
        raise ValueError('Retry count exceeded when expecting: {0}'.format(expected))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True, choices=POLICIES)
    parser.add_argument('--encoding', type=str, required=True, choices=['standard', 'group'])
    parser.add_argument('--encryption', type=str, required=True, choices=['block', 'stream'])
    parser.add_argument('--collection-rate', type=float, required=True, nargs='+')
    parser.add_argument('--should-compress', action='store_true')
    parser.add_argument('--max-num-samples', type=int)
    parser.add_argument('--should-print', action='store_true')
    args = parser.parse_args()

    # Unpack the target collection rates
    assert len(args.collection_rate) in (1, 3), 'Must provide 1 rate or a range of collection rates'

    if len(args.collection_rate) == 1:
        collection_rates = args.collection_rate
    else:
        collection_rates = np.arange(start=args.collection_rate[0], stop=args.collection_rate[1] + 1e-5, step=args.collection_rate[2]).tolist()

    # Make the output folder
    current_date = datetime.now().strftime('%Y-%m-%d')
    base = os.path.join('saved_models', args.dataset, current_date)
    make_dir(base)

    folder_name = '{0}_{1}'.format(args.policy, args.encoding)
    output_folder = os.path.join(base, folder_name)
    make_dir(output_folder)

    for collection_rate in sorted(collection_rates):

        collection_rate = round(collection_rate, 2)

        if args.should_print:
            print('==========')
            print('Starting {0:.2f}'.format(collection_rate))
            print('==========')

        port = random.randint(50000, 60000)

        # Set the commands
        if args.max_num_samples is None:
            server_cmd = SERVER_CMD_ALL.format(args.dataset, args.encryption, args.policy, args.encoding, collection_rate, output_folder, port)
            sensor_cmd = SENSOR_CMD_ALL.format(args.dataset, args.encryption, args.policy, args.encoding, collection_rate, port)
        else:
            server_cmd = SERVER_CMD_SAMPLES.format(args.dataset, args.encryption, args.policy, args.encoding, collection_rate, output_folder, port, args.max_num_samples)
            sensor_cmd = SENSOR_CMD_SAMPLES.format(args.dataset, args.encryption, args.policy, args.encoding, collection_rate, port, args.max_num_samples)

        if args.should_compress:
            server_cmd += ' --should-compress'
            sensor_cmd += ' --should-compress'

        server, sensor = None, None

        try:
            # Start the server
            server = pexpect.spawn(server_cmd)

            expect_with_retry(comm_module=server, expected='Started Server.')

            # Start the sensor
            sensor = pexpect.spawn(sensor_cmd)
            
            expect_with_retry(comm_module=server, expected='Accepted connection')

            # Print out progress
            for line in server:
                progress = line.decode().strip()
                if progress.startswith('Completed') and args.should_print:
                    print(progress, end='\r')

            if args.should_print:
                print()

            # Wait for completion
            sensor.expect('Completed Sensor.')

            sensor.expect(pexpect.EOF)
            server.expect(pexpect.EOF)
        finally:
            if server is not None:
                server.close()

            if sensor is not None:
                sensor.close()
