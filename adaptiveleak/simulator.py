import os
import pexpect
import sys
import random
from argparse import ArgumentParser
from datetime import datetime
from typing import List

from adaptiveleak.utils.constants import POLICIES
from adaptiveleak.utils.file_utils import make_dir


SERVER_CMD_ALL = 'python server.py --dataset {0} --encryption {1} --params {2} --output-folder {3} --port {4}'
SERVER_CMD_SAMPLES = 'python server.py --dataset {0} --encryption {1} --params {2} --output-folder {3} --port {4} --max-num-samples {5}'

SENSOR_CMD_ALL = 'python sensor.py --dataset {0} --encryption {1} --params {2} --port {3}'
SENSOR_CMD_SAMPLES = 'python sensor.py --dataset {0} --encryption {1} --params {2} --port {3} --max-num-samples {4}'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True, choices=POLICIES)
    parser.add_argument('--policy-params', type=str, required=True, nargs='+')
    parser.add_argument('--encryption', type=str, required=True, choices=['block', 'stream'])
    parser.add_argument('--should-compress', action='store_true')
    parser.add_argument('--max-num-samples', type=int)
    args = parser.parse_args()

    # Unpack the parameter files by detecting directories
    param_files: List[str] = []
    for param_file in args.policy_params:
        if os.path.isdir(param_file):
            file_names = [name for name in os.listdir(param_file) if name.endswith('.json')]
            param_files.extend(os.path.join(param_file, name) for name in file_names)
        else:
            param_files.append(param_file)

    # Make the output folder
    current_date = datetime.now().strftime('%Y-%m-%d')
    base = os.path.join('saved_models', args.dataset, current_date)
    make_dir(base)

    output_folder = os.path.join(base, args.policy)
    make_dir(output_folder)

    for params_path in sorted(param_files):

        print('==========')
        print('Starting {0}'.format(params_path))
        print('==========')

        port = random.randint(50000, 60000)

        # Set the commands
        if args.max_num_samples is None:
            server_cmd = SERVER_CMD_ALL.format(args.dataset, args.encryption, params_path, output_folder, port)
            sensor_cmd = SENSOR_CMD_ALL.format(args.dataset, args.encryption, params_path, port)
        else:
            server_cmd = SERVER_CMD_SAMPLES.format(args.dataset, args.encryption, params_path, output_folder, port, args.max_num_samples)
            sensor_cmd = SENSOR_CMD_SAMPLES.format(args.dataset, args.encryption, params_path, port, args.max_num_samples)

        if args.should_compress:
            server_cmd += ' --should-compress'
            sensor_cmd += ' --should-compress'

        print(server_cmd)

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
