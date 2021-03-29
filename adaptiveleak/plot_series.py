import matplotlib.pyplot as plt
import numpy as np
import h5py
import os.path
from argparse import ArgumentParser
from sklearn.metrics import mean_squared_error
from typing import List, Tuple

from policies import make_policy, Policy
from rounding import quantize
from transition_model import LinearModel
from utils.constants import LINEAR_TRANSITION
from utils.file_utils import read_pickle_gz


def execute_policy(policy: Policy, sequence: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    
    estimate_list: List[np.ndarray] = []
    collected_list: List[np.ndarray] = []
    collected: List[int] = []
    
    policy.reset()

    for idx in range(len(sequence)):
        # policy.transition()

        measurement = np.expand_dims(sequence[idx], axis=-1)
        did_send = policy.transmit(measurement=measurement, seq_idx=idx)

        estimate = policy.get_estimate().reshape(1, -1)  # [1, D]

        if did_send:
            collected.append(idx)
            collected_list.append(estimate)

    transmitted_seq, total_bytes = quantize(measurements=np.vstack(collected_list),
                                            num_transmitted=len(collected_list),
                                            policy=policy,
                                            should_pad=False)

    estimate = np.zeros_like(collected_list[0])
    collected_idx = 0

    for idx in range(len(sequence)):
        if (collected_idx < len(collected)) and (idx == collected[collected_idx]):
            estimate = transmitted_seq[collected_idx].reshape(1, -1)
            collected_idx += 1

        estimate_list.append(estimate)

    return np.vstack(estimate_list), collected


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--feature', type=int, default=0)
    args = parser.parse_args()

    data_file = os.path.join('datasets', args.dataset, 'train', 'data.h5')
    with h5py.File(data_file, 'r') as fin:
        series = fin['inputs'][:]
        label = fin['output'][:]

    if len(series.shape) == 2:
        series = np.expand_dims(series, axis=-1)

    # Scale the input
    input_shape = series.shape
    scaler = read_pickle_gz(os.path.join('saved_models', args.dataset, 'mlp_scaler.pkl.gz'))
    series = scaler.transform(series.reshape(-1, input_shape[-1])).reshape(input_shape)

    # Collect the transition model
    transition_path = os.path.join('saved_models', args.dataset, '{0}.pkl.gz'.format(LINEAR_TRANSITION))
    transition_model = LinearModel.restore(transition_path)

    # Make the policy
    policy = make_policy(name=args.policy,
                         transition_model=transition_model,
                         seq_length=series.shape[1],
                         num_features=series.shape[2],
                         target=0.5,
                         threshold=0.0,
                         width=8,
                         precision=6,
                         use_confidence=False,
                         compression_name='block',
                         compression_params=dict())

    errors: List[float] = []
    estimate_list: List[np.ndarray] = []
    collected: List[List[int]] = []
    for idx, inputs in enumerate(series):
        estimates, collected_idx = execute_policy(policy=policy, sequence=inputs)
        policy.step(seq_idx=idx, count=len(collected_idx))
        error = mean_squared_error(y_true=inputs, y_pred=estimates)

        errors.append(error)
        collected.append(collected_idx)
        estimate_list.append(estimates)

    num_samples = input_shape[0] * input_shape[1]
    num_collected = sum(len(c) for c in collected)

    print('MSE: {0:.5f}'.format(np.average(errors)))
    print('Number Collected: {0} / {1} ({2:.4f})'.format(num_collected, num_samples, num_collected / num_samples))

    data_idx = np.argmax(errors)
    estimates = estimate_list[data_idx]
    collected_idx = collected[data_idx]

    print('Max Error: {0:.5f} (Idx: {1})'.format(errors[data_idx], data_idx))
    print('Max Error Collected: {0}'.format(len(collected_idx)))

    with plt.style.context('seaborn-ticks'):
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

        xs = list(range(input_shape[1]))
        ax1.plot(xs, series[data_idx, :, args.feature])
        ax1.plot(xs, estimates[:, args.feature])
        ax1.scatter(collected_idx, estimates[collected_idx, args.feature], marker='o')

        ax2.hist(x=errors, bins=50)

        plt.show()
