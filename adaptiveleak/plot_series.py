import matplotlib.pyplot as plt
import numpy as np
import h5py
import os.path
from argparse import ArgumentParser
from sklearn.metrics import mean_squared_error
from typing import List, Tuple

from adaptiveleak.server import reconstruct_sequence
from adaptiveleak.policies import make_policy, Policy, run_policy
from adaptiveleak.utils.encryption import EncryptionMode
from adaptiveleak.utils.file_utils import read_pickle_gz


def execute_policy(policy: Policy, sequence: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    
    estimate_list: List[np.ndarray] = []
    collected_list: List[np.ndarray] = []
    collected: List[int] = []
    
    policy.reset()

    for idx in range(len(sequence)):
        # policy.transition()

        measurement = np.expand_dims(sequence[idx], axis=-1)
        did_send = policy.should_collect(measurement=measurement, seq_idx=idx)

        estimate = policy.get_estimate().reshape(1, -1)  # [1, D]

        if did_send:
            policy.collect()

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
        inputs = fin['inputs'][:]
        label = fin['output'][:]

    if len(inputs.shape) == 2:
        inputs = np.expand_dims(inputs, axis=-1)

    # Unpack the shape
    num_seq, seq_length, num_features = inputs.shape

    # Get any existing thresholds
    thresholds_path = os.path.join('saved_models', args.dataset, 'thresholds.pkl.gz')
    thresholds = read_pickle_gz(thresholds_path)

    # Make the policy
    target = 0.5
    policy = make_policy(name=args.policy,
                         seq_length=seq_length,
                         num_features=num_features,
                         target=target,
                         threshold=thresholds.get(target, 0.0),
                         width=8,
                         precision=6,
                         encryption_mode=EncryptionMode.STREAM,
                         encoding='standard')

    errors: List[float] = []
    estimate_list: List[np.ndarray] = []
    collected: List[List[int]] = []

    collected_seq = min(num_seq, 500)

    for idx, sequence in enumerate(inputs):
        if idx >= collected_seq:
            break

        policy.reset()

        estimates, collected_idx = run_policy(policy=policy, sequence=sequence)
        policy.step(seq_idx=idx, count=len(collected_idx))

        reconstructed = reconstruct_sequence(measurements=estimates, collected_indices=collected_idx, seq_length=seq_length)

        error = mean_squared_error(y_true=sequence, y_pred=reconstructed)

        errors.append(error)
        collected.append(collected_idx)
        estimate_list.append(reconstructed)

    num_samples = collected_seq * seq_length
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

        xs = list(range(seq_length))
        ax1.plot(xs, inputs[data_idx, :, args.feature])
        ax1.plot(xs, estimates[:, args.feature])
        ax1.scatter(collected_idx, estimates[collected_idx, args.feature], marker='o')

        ax2.hist(x=errors, bins=50)

        plt.show()
