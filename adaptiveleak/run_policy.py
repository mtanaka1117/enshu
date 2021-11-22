import matplotlib.pyplot as plt
import numpy as np
import os.path
import time
from argparse import ArgumentParser
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from typing import List, Tuple

from adaptiveleak.server import reconstruct_sequence
from adaptiveleak.policies import BudgetWrappedPolicy, Policy, run_policy
from adaptiveleak.utils.analysis import normalized_mae, normalized_rmse
from adaptiveleak.utils.constants import ENCODING
from adaptiveleak.utils.file_utils import read_pickle_gz, save_pickle_gz
from adaptiveleak.utils.loading import load_data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--collection-rate', type=float, required=True)
    parser.add_argument('--encoding', type=str, choices=ENCODING, default='standard')
    parser.add_argument('--feature', type=int, default=0)
    parser.add_argument('--max-num-samples', type=int)
    args = parser.parse_args()

    # Load the data
    fold = 'test'
    inputs, labels = load_data(dataset_name=args.dataset, fold=fold)

    labels = labels.reshape(-1)

    # Unpack the shape
    num_seq, seq_length, num_features = inputs.shape

    # Make the policy
    collection_rate = args.collection_rate

    policy = BudgetWrappedPolicy(name=args.policy,
                                 seq_length=seq_length,
                                 num_features=num_features,
                                 dataset=args.dataset,
                                 collection_rate=args.collection_rate,
                                 encryption_mode='stream',
                                 collect_mode='tiny',
                                 encoding=args.encoding,
                                 should_compress=False)

    energy_list: List[float] = []
    bytes_list: List[int] = []
    errors: List[float] = []
    estimate_list: List[np.ndarray] = []
    collected: List[List[int]] = []
    collected_counts = defaultdict(list)

    max_num_seq = num_seq if args.max_num_samples is None else min(num_seq, args.max_num_samples)

    policy.init_for_experiment(num_sequences=max_num_seq)

    collected_seq = 1  # The number of sequences collected under the budget

    for idx, (sequence, label) in enumerate(zip(inputs, labels)):
        if idx >= max_num_seq:
            break

        policy.reset()

        # Run the policy
        policy_result = run_policy(policy=policy,
                                   sequence=sequence,
                                   should_enforce_budget=False)
        policy.step(seq_idx=idx, count=policy_result.num_collected)

        # Decode the sequence (accounts for numerical errors)
        if policy_result.num_bytes > 0:
            recv_measurements, recv_indices, _ = policy.decode(message=policy_result.encoded)
        else:
            recv_measurements = policy_result.measurements
            recv_indices = policy_result.collected_indices

        # Reconstruct the sequence
        reconstructed = reconstruct_sequence(measurements=recv_measurements,
                                             collected_indices=recv_indices,
                                             seq_length=seq_length)

        error = mean_absolute_error(y_true=sequence, y_pred=reconstructed)

        # Record the policy results
        errors.append(error)
        estimate_list.append(reconstructed)
        collected.append(policy_result.collected_indices)
        collected_counts[label].append(policy_result.num_bytes)
        energy_list.append(policy_result.energy)
        bytes_list.append(policy_result.num_bytes)

        if not policy.has_exhausted_budget():
            collected_seq = idx + 1

    num_samples = collected_seq * seq_length
    num_collected = sum(len(c) for c in collected[:collected_seq])

    reconstructed = np.vstack([np.expand_dims(r, axis=0) for r in estimate_list])  # [N, T, D]
    reconstructed = reconstructed.reshape(-1, num_features)

    true = inputs[0:max_num_seq]
    true = true.reshape(-1, num_features)

    error = mean_absolute_error(y_true=true, y_pred=reconstructed)
    norm_error = normalized_mae(y_true=true, y_pred=reconstructed)

    rmse = mean_squared_error(y_true=true, y_pred=reconstructed, squared=False)
    norm_rmse = normalized_rmse(y_true=true, y_pred=reconstructed)

    r2 = r2_score(y_true=true, y_pred=reconstructed, multioutput='variance_weighted')

    print('MAE: {0:.7f}, Norm MAE: {1:.5f}, RMSE: {2:.5f}, Norm RMSE: {3:.5f}, R^2: {4:.5f}'.format(error, norm_error, rmse, norm_rmse, r2))
    print('Number Collected: {0} / {1} ({2:.4f})'.format(num_collected, num_samples, num_collected / num_samples))
    print('Energy: {0:.5f} mJ (Budget: {1:.5f})'.format(policy.consumed_energy, policy.budget))
    print('Energy Per Seq: {0:.5f} (Budget: {1:.5f})'.format(np.average(energy_list[:collected_seq]), policy.energy_per_seq))
    print('Byte Count: {0:.5f} ({1:.5f})'.format(np.average(bytes_list[:collected_seq]), np.std(bytes_list[:collected_seq])))
    print('Collected: {0} / {1}'.format(collected_seq, max_num_seq))

    data_idx = np.argmax(errors)
    estimates = estimate_list[data_idx]
    collected_idx = collected[data_idx]

    print('Max Error: {0:.5f} (Idx: {1})'.format(errors[data_idx], data_idx))
    print('Max Error Collected: {0}'.format(len(collected_idx)))

    print('Label Distribution')
    for label, counts in collected_counts.items():
        print('{0} -> {1:.2f} ({2:.2f})'.format(label, np.average(counts), np.std(counts)))

    with plt.style.context('seaborn-ticks'):
        fig, ax1 = plt.subplots()

        xs = list(range(seq_length))
        ax1.plot(xs, inputs[data_idx, :, args.feature], label='True')
        ax1.plot(xs, estimates[:, args.feature], label='Inferred')
        ax1.scatter(collected_idx, estimates[collected_idx, args.feature], marker='o')

        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Feature Value')

        ax1.legend()

        plt.show()
