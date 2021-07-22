import numpy as np
import os
import sys
import time
from argparse import ArgumentParser
from functools import reduce
from typing import Optional, Iterable, List

from ble_manager import BLEManager


MAC_ADDRESS = '00:35:FF:13:A3:1E'
BLE_HANDLE = 18
HCI_DEVICE = 'hci0'
SLEEP = 2


def execute_client(num_bytes: int, num_trials: int):
    """
    Starts the device client. This function connects with the devices (to reset the sleep mode), and then recieves
    a single message when specified.

    Args:
        num_bytes: The number of bytes to receive
    """
    assert num_bytes >= 1, 'Must provide a positive number of bytes'

    # Initialize the device manager
    device_manager = BLEManager(mac_addr=MAC_ADDRESS, handle=BLE_HANDLE, hci_device=HCI_DEVICE)

    print('==========')
    print('Starting experiment')
    print('==========')

    # Start and reset the device
    try:
        device_manager.start()
        #device_manager.send_and_expect_byte(value=b'\xab', expected='\xcd')
    finally:
        device_manager.stop()

    print('Press any key to start...')
    x = input()

    for _ in range(num_trials):
        try:
            device_manager.start()
            response = device_manager.query(value=b'\xab')
            #device_manager.send(value=b'\xab')

            #data = num_bytes.to_bytes(2, 'big')
            #response = device_manager.query(value=data)
            print('Received response of {0} bytes (Target: {1})'.format(len(response), num_bytes))
        finally:
            device_manager.stop()

        time.sleep(SLEEP)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num-bytes', type=int, required=True)
    parser.add_argument('--num-trials', type=int, required=True)
    args = parser.parse_args()

    execute_client(num_bytes=args.num_bytes, num_trials=args.num_trials)
