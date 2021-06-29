import h5py
import os.path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple

from adaptiveleak.utils.data_utils import array_to_fp, array_to_fp_unsigned, array_to_float, select_range_shift, to_fixed_point, to_float
from adaptiveleak.utils.shifting import merge_shift_groups


NUM_SAMPLES = 1
WIDTH = 9
PRECISIONS = list(range(0, WIDTH))
DATASET_NAME = 'mnist'

with h5py.File(os.path.join('..', 'datasets', DATASET_NAME, 'train', 'data.h5'), 'r') as fin:
    inputs = fin['inputs'][:NUM_SAMPLES]
    output = fin['output'][:NUM_SAMPLES]

inputs = inputs.reshape(-1, inputs.shape[-1])

maes: List[float] = []
mapes: List[float] = []

for precision in PRECISIONS:
    quantized = array_to_fp(inputs, width=WIDTH, precision=precision)
    recovered = array_to_float(quantized, precision=precision)
    #recovered = inputs.astype(np.float16)

    mae = mean_absolute_error(y_true=inputs, y_pred=recovered)
    maes.append(mae)

    mape = mean_absolute_percentage_error(y_true=inputs, y_pred=recovered)
    mapes.append(mape)

mae_idx = np.argmin(maes)
min_mae = maes[mae_idx]
best_mae_precision = PRECISIONS[mae_idx]

mape_idx = np.argmin(mapes)
min_mape = mapes[mape_idx]
best_mape_precision = PRECISIONS[mape_idx]

print('{0} -> Min MAE: {1} ({2}), Min MAPE: {3} ({4})'.format(DATASET_NAME, min_mae, best_mae_precision, min_mape, best_mape_precision))

with plt.style.context('seaborn-ticks'):
    fig, ax = plt.subplots()

    ax.plot(PRECISIONS, maes, marker='o', label='MAE')
    ax.plot(PRECISIONS, mapes, marker='o', label='MAPE')

    ax.legend()

    plt.show()
    

# Scaled
#scaled_fp = array_to_fp_unsigned(scaled_inputs, width=WIDTH, precision=WIDTH - 1)
#scaled_quantized = array_to_float(scaled_fp, precision=WIDTH - 1)
#
#scaled_quantized *= (max_elements - min_elements)
#scaled_quantized += min_elements
#
#error_scaled = mean_absolute_error(y_true=inputs, y_pred=scaled_quantized)
#
#print('Scaled Error: {0}'.format(error_scaled))
