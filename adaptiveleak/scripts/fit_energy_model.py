import os.path
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics import r2_score, mean_absolute_error
from typing import List

from adaptiveleak.utils.file_utils import save_json, iterate_dir, read_json


def get_energy(folder: str) -> float:
    energy_path = os.path.join(folder, 'energy.json')
    energy_dict = read_json(energy_path)
    return float(np.median(energy_dict['energy']))


def plot(num_bytes: np.ndarray, energy: np.ndarray, energy_pred: np.ndarray, output_path: str):
    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots()

        ax.scatter(num_bytes, energy, marker='o', label='True')
        ax.plot(num_bytes, energy_pred, label='Pred')

        ax.set_xlabel('Num Bytes')
        ax.set_ylabel('Energy (mJ)')
        ax.set_title('Energy (mJ) per Operation for Multiple Sizes')

        plt.savefig(output_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    args = parser.parse_args()

    base = args.folder

    bytes_list: List[int] = []
    energy_list: List[float] = []

    # Get the energy for each trace folder
    for trace_folder in iterate_dir(base, pattern='.*'):
        name = os.path.split(trace_folder)[-1]

        try:
            num_bytes = int(name)
            energy = get_energy(folder=trace_folder)

            bytes_list.append([[num_bytes, 1]])
            energy_list.append(energy)
        except ValueError:
            continue

    X = np.vstack(bytes_list)  # [N, 2]
    y = np.vstack(energy_list)  # [N]
    reg = 0.01 * np.eye(X.shape[-1])

    weights = np.linalg.solve(X.T.dot(X) + reg, X.T.dot(y))  # [2]
    
    w = weights[0]
    b = weights[1]

    ypred = X.dot(weights)  # [N]

    plot(num_bytes=X[:, 0],
         energy=y,
         energy_pred=ypred,
         output_path=os.path.join(base, 'fit.pdf'))

    result = {
        'w': float(w),
        'b': float(b)
    }

    save_json(result, os.path.join(base, 'model.json'))
