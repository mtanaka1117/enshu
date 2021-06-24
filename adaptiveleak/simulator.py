import os
import pexpect
import sys
import random
import numpy as np
from argparse import ArgumentParser
from datetime import datetime
from typing import List

from adaptiveleak.utils.constants import POLICIES
from adaptiveleak.utils.file_utils import make_dir


SERVER_CMD_ALL = 'python server.py --dataset {0} --encryption {1} --policy {2} --encoding {3} --target {4} --output-folder {5} --port {6}'
SERVER_CMD_SAMPLES = 'python server.py --dataset {0} --encryption {1} --policy {2} --encoding {3} --target {4} --output-folder {5} --port {6} --max-num-samples {7}'

SENSOR_CMD_ALL = 'python sensor.py --dataset {0} --encryption {1} --policy {2} --encoding {3} --target {4} --port {5}'
SENSOR_CMD_SAMPLES = 'python sensor.py --dataset {0} --encryption {1} --policy {2} --encoding {3} --target {4} --port {5} --max-num-samples {6}'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True, choices=POLICIES)
    parser.add_argument('--encoding', type=str, required=True, choices=['standard', 'group'])
    parser.add_argument('--encryption', type=str, required=True, choices=['block', 'stream'])
    parser.add_argument('--target', type=float, required=True, nargs='+')
    parser.add_argument('--should-compress', action='store_true')
    parser.add_argument('--max-num-samples', type=int)
    args = parser.parse_args()

    # Unpack the targets
    assert len(args.target) in (1, 3), 'Must provide 1 target or a range of targets'

    if len(args.target) == 1:
        targets = args.target
    else:
        targets = np.arange(start=args.target[0], stop=args.target[1] + 1e-5, step=args.target[2]).tolist()

    # Make the output folder
    current_date = datetime.now().strftime('%Y-%m-%d')
    base = os.path.join('saved_models', args.dataset, current_date)
    make_dir(base)

    folder_name = '{0}_{1}'.format(args.policy, args.encoding)
    output_folder = os.path.join(base, folder_name)
    make_dir(output_folder)

    for target in sorted(targets):

        target = round(target, 2)

        print('==========')
        print('Starting {0:.2f}'.format(target))
        print('==========')

        port = random.randint(50000, 60000)

        # Set the commands
        if args.max_num_samples is None:
            server_cmd = SERVER_CMD_ALL.format(args.dataset, args.encryption, args.policy, args.encoding, target, output_folder, port)
            sensor_cmd = SENSOR_CMD_ALL.format(args.dataset, args.encryption, args.policy, args.encoding, target,  port)
        else:
            server_cmd = SERVER_CMD_SAMPLES.format(args.dataset, args.encryption, args.policy, args.encoding, target, output_folder, port, args.max_num_samples)
            sensor_cmd = SENSOR_CMD_SAMPLES.format(args.dataset, args.encryption, args.policy, args.encoding, target, port, args.max_num_samples)

        if args.should_compress:
            server_cmd += ' --should-compress'
            sensor_cmd += ' --should-compress'

        server, sensor = None, None

        try:
            # Start the server
            server = pexpect.spawn(server_cmd)
            server.expect('Started Server.')

            # Start the sensor
            sensor = pexpect.spawn(sensor_cmd)

            server.expect('Accepted connection', timeout=5)

            # Print out progress
            for line in server:
                progress = line.decode().strip()
                if progress.startswith('Completed'):
                    print(progress, end='\r')

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
