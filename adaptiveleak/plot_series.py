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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--target', type=float, required=True)
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

    # Make the policy
    target = args.target

    policy = make_policy(name=args.policy,
                         seq_length=seq_length,
                         num_features=num_features,
                         dataset=args.dataset,
                         target=target,
                         encryption_mode=EncryptionMode.STREAM,
                         encoding='standard')

    errors: List[float] = []
    estimate_list: List[np.ndarray] = []
    collected: List[List[int]] = []

    # collected_seq = min(num_seq, 400)
    collected_seq = num_seq

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
