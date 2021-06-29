import h5py
import os.path
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_absolute_error
from typing import List, Tuple

from adaptiveleak.utils.data_utils import array_to_fp_shifted, array_to_float_shifted, select_range_shifts_array, to_fixed_point, to_float
from adaptiveleak.utils.shifting import merge_shift_groups, array_to_float, array_to_fp


NUM_SAMPLES = 100


def get_errors(dataset_name: str, widths: List[int], non_fractional: int) -> List[float]:

    with h5py.File(os.path.join('..', 'datasets', dataset_name, 'train', 'data.h5'), 'r') as fin:
        inputs = fin['inputs'][:NUM_SAMPLES]
        output = fin['output'][:NUM_SAMPLES]

    shifted_errors: List[float] = []
    unshifted_errors: List[float] = []
    runtimes: List[float] = []

    for width in widths:

        precision = width - non_fractional

        shifted_quantized: List[float] = []
        shifted_features: List[float] = []
        true_features: List[float] = []

        for features in inputs:
            flattened = features.T.reshape(-1).astype(float)

            shifts = select_range_shifts_array(flattened,
                                               width=width,
                                               precision=precision,
                                               num_range_bits=3)

            start = time.perf_counter()
            merged_shifts, reps = merge_shift_groups(values=flattened,
                                                     shifts=shifts,
                                                     max_num_groups=6)
            end = time.perf_counter()

            runtimes.append(end - start)

            shifts = np.repeat(merged_shifts, reps)

            fixed_point = array_to_fp_shifted(flattened, width=width, precision=precision, shifts=shifts)
            recovered = array_to_float_shifted(fixed_point, precision=precision, shifts=shifts)

            shifted_quantized.extend(recovered)
            shifted_features.extend(flattened)
            true_features.extend(features)

        shifted_error = mean_absolute_error(y_true=shifted_features, y_pred=shifted_quantized)

        fixed_point = array_to_fp(np.array(true_features), width=width, precision=precision)
        quantized = array_to_float(fixed_point, precision=precision)
        unshifted_error = mean_absolute_error(y_true=true_features, y_pred=quantized)
        
        shifted_errors.append(shifted_error)
        unshifted_errors.append(unshifted_error)

    print('Median Runtime: {0:.6f}, Max Runtime: {1:.6f}'.format(np.median(runtimes), np.max(runtimes)))

    return shifted_errors, unshifted_errors


if __name__ == '__main__':
    dataset_name = 'uci_har'
    widths = [4, 6, 8, 10, 12, 16]
    non_fractional = 3

    shifted_errors, unshifted_errors = get_errors(dataset_name, widths, non_fractional)
    print(shifted_errors)
    print(unshifted_errors)

    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots()

        ax.plot(widths, shifted_errors, label='Dynamic Range', marker='o')
        ax.plot(widths, unshifted_errors, label='Fixed Range', marker='o')

        ax.set_title('Quantization Error for {0} Dataset'.format(dataset_name.capitalize()))
        ax.set_xlabel('Bit Width')
        ax.set_ylabel('Mean Absolute Error')
        ax.legend()

        plt.show()
