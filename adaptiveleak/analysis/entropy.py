import h5py
import os.path
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from adaptiveleak.analysis.plot_utils import dataset_label


NUM_SAMPLES = 2500
COLORS = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#abd9e9', '#74add1', '#4575b4']

def compute_entropy(bins: List[int], dataset_name: str) -> List[float]:
    with h5py.File(os.path.join('..', 'datasets', dataset_name, 'train', 'data.h5'), 'r') as fin:
        inputs = fin['inputs'][:NUM_SAMPLES]

    flattened = inputs.reshape(-1)
    results: List[float] = []

    for bin_count in bins:
        # Compute the data histogram
        hist, _ = np.histogram(flattened, bins=bin_count)

        # Normalize the counts
        freq = hist / np.sum(hist)

        # Compute the entropy
        entropy = np.sum(-1 * freq * np.log2(freq + 1e-7))

        # Save the entropy values
        results.append(entropy)

    return results


if __name__ == '__main__':
    datasets = ['epilepsy', 'haptics', 'uci_har', 'pavement', 'trajectories', 'strawberry', 'tiselac']
    bins = [16, 32, 64, 128, 256, 512, 1024]

    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots()

        for i, dataset_name in enumerate(sorted(datasets)):
            entropy = compute_entropy(bins=bins, dataset_name=dataset_name)
            ax.plot(bins, entropy, label=dataset_label(dataset_name), linewidth=2, marker='o', color=COLORS[i])

        # Include the 'uniform' encoding amounts
        ax.plot(bins, [np.log2(b) for b in bins], label='Uniform', linewidth=2, marker='o', color='k')

        ax.set_title('Measurement Value Entropy for Various Quantization Levels')
        ax.set_ylabel('Entropy (bits)')
        ax.set_xlabel('Number of Quantization Buckets')

        ax.legend()

        plt.show()
