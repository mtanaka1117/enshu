import os.path
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics import r2_score, mean_absolute_error
from typing import List, Tuple

from adaptiveleak.utils.file_utils import save_json, iterate_dir, read_json
from adaptiveleak.utils.constants import BT_FRAME_SIZE
from adaptiveleak.utils.data_utils import round_to_block


def get_result_field(folder: str, field: str) -> float:
    energy_path = os.path.join(folder, 'energy.json')
    energy_dict = read_json(energy_path)
    return float(np.median(energy_dict[field]))


def plot(num_bytes: np.ndarray, true: np.ndarray, pred: np.ndarray, output_path: str):
    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots()

        ax.scatter(num_bytes, true, marker='o', label='True')
        ax.plot(num_bytes, pred, label='Pred')

        ax.set_xlabel('Num Bytes')
        ax.set_ylabel('Energy (mJ)')
        ax.set_title('{0} per Operation for Various Message Sizes'.format('Energy'))

        plt.savefig(output_path)


def fit(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    reg = 0.01 * np.eye(X.shape[-1])

    weights = np.linalg.solve(X.T.dot(X) + reg, X.T.dot(y))  # [2]
    
    w = weights[0]
    b = weights[1]

    return w, b


def evaluate_model(weights: np.ndarray, field: str, folder: str) -> float:
    bytes_list: List[List[int]] = []
    y_list: List[float] = []

    for trace_folder in iterate_dir(folder, pattern='.*'):
        name = os.path.split(trace_folder)[-1]
        
        try:
            num_bytes = int(name)

            if num_bytes % BT_FRAME_SIZE != 0:
                num_bytes = round_to_block(num_bytes, BT_FRAME_SIZE)

        except ValueError:
            continue

        y = get_result_field(folder=trace_folder, field=field)
        
        bytes_list.append([[num_bytes, 1]])
        y_list.append(y)

    num_bytes = np.vstack(bytes_list)
    ys = np.vstack(y_list)

    pred = num_bytes.dot(weights)
    error = np.average(np.abs(pred - ys))

    return float(error)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    args = parser.parse_args()

    base = args.folder

    bytes_list: List[int] = []
    comm_energy_list: List[float] = []
    #time_list: List[float] = []
    baseline_list: List[float] = []

    # Get the energy for each trace folder
    for trace_folder in iterate_dir(base, pattern='.*'):
        name = os.path.split(trace_folder)[-1]

        if name == 'validation':
            continue

        try:
            num_bytes = int(name)
            comm_energy = get_result_field(folder=trace_folder, field='comm_energy')
            baseline_power = get_result_field(folder=trace_folder, field='baseline_power')

            bytes_list.append([[num_bytes, 1]])
            comm_energy_list.append(comm_energy)
            baseline_list.append(baseline_power)
        except ValueError:
            continue

    comm_energy = np.vstack(comm_energy_list)
    baseline_power = np.vstack(baseline_list)
    num_bytes = np.vstack(bytes_list)

    comm_weights = fit(X=num_bytes, y=comm_energy)
    base_weights = fit(X=num_bytes, y=baseline_power)

    pred_comm_energy = num_bytes.dot(comm_weights)
    pred_base_power = num_bytes.dot(base_weights)

    plot(num_bytes=num_bytes[:, 0],
         true=comm_energy,
         pred=pred_comm_energy,
         output_path=os.path.join(base, 'comm_energy.pdf'))

    plot(num_bytes=num_bytes[:, 0],
         true=baseline_power,
         pred=pred_base_power,
         output_path=os.path.join(base, 'base_energy.pdf'))

    # Validate the model
    train_comm_error = evaluate_model(weights=comm_weights, field='comm_energy', folder=base)
    train_base_error = evaluate_model(weights=base_weights, field='baseline_power', folder=base)

    validation_comm_error = evaluate_model(weights=comm_weights, field='comm_energy', folder=os.path.join(base, 'validation'))
    validation_base_error = evaluate_model(weights=base_weights, field='baseline_power', folder=os.path.join(base, 'validation'))
    

    result = {
        'comm_w': float(comm_weights[0]),
        'comm_b': float(comm_weights[1]),
        'base_w': float(base_weights[0]),
        'base_b': float(base_weights[1]),
        'train_comm_error': train_comm_error,
        'train_base_error': train_base_error,
        'val_comm_error': validation_comm_error,
        'val_base_error': validation_base_error
    }

    save_json(result, os.path.join(base, 'model.json'))
