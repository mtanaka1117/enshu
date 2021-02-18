import numpy as np
import h5py
import os
from argparse import ArgumentParser
from collections import defaultdict
from sklearn.metrics import accuracy_score
from typing import Optional, List, Tuple, Dict, Any

from policies import Policy, make_policy
from server import Server
from utils.file_utils import read_pickle_gz, save_pickle_gz, read_json
from utils.data_utils import  calculate_bytes


X_OFFSET = 0.2
Y_OFFSET = 0.01
PRINT_FREQ = 100


def run(inputs: np.ndarray,
        output: np.ndarray,
        policy: Policy,
        server: Server,
        policy_params: Dict[str, Any],
        max_num_samples: Optional[int],
        output_folder: str):
    rand = np.random.RandomState(seed=9389)

    # Shuffle the data (in a consistent manner)
    data_idx = np.arange(inputs.shape[0])
    rand.shuffle(data_idx)

    inputs = inputs[data_idx]
    output = output[data_idx]

    # Get the number of samples to evaluate
    num_samples = inputs.shape[0] if max_num_samples is None else max_num_samples
    inputs = inputs[:num_samples]  # [N, T, D]
    output = output[:num_samples]  # [N]

    # Unpack the shape
    seq_length, num_features = inputs.shape[1], inputs.shape[2]

    # Normalize all of the data first
    scaled = server.scale(inputs.reshape(-1, num_features))
    inputs = scaled.reshape(-1, seq_length, num_features)
    output = output.reshape(-1)

    # Count the number of transmitted values per class
    transmit_dist: DefaultDict[int, List[int]] = defaultdict(list)
    size_dist: DefaultDict[int, List[int]] = defaultdict(list)

    # Store the measurements captured by the server
    server_recieved: List[int] = []

    # Count the fraction of transmitted measurements
    total_transmitted = 0

    for sample_idx in range(num_samples):
        seq_features = inputs[sample_idx]  # [T, D]

        seq_list: List[np.ndarray] = []
        sent_idx: List[int] = []

        # Reset the policy before every sequence
        policy.reset()

        num_transmitted = 0
        for seq_idx, input_features in enumerate(seq_features):
            # Update the estimate
            policy.transition()

            # Determine whether to transmit the measurement
            measurement = np.expand_dims(input_features, axis=-1)  # [D, 1]

            did_send = policy.transmit(measurement=measurement, seq_idx=seq_idx)
            num_transmitted += did_send

            estimate = policy.get_estimate().reshape(1, -1)  # [1, D]

            if did_send:
                sent_idx.append(seq_idx)

            # Save the estimates for later analysis
            seq_list.append(estimate)

        # Always send at least one measurement
        if num_transmitted == 0:
            seq_list.append(seq_features[0].reshape(1, -1))
            sent_idx.append(0)
            num_transmitted += 1

        seq_measurements = np.vstack(seq_list)  # [T, D]

        # Translate measurements to fixed-point representation (and back)
        transmitted_seq, total_bytes = policy.quantize_seq(seq_measurements,
                                                           num_transmitted=num_transmitted)

        # print('Num Transmitted: {0}, Total Bytes: {1}'.format(num_transmitted, total_bytes))

        # Simulate sending the (un-scaled) measurements to the server
        sent_measurements = [np.expand_dims(transmitted_seq[i], axis=0) for i in sent_idx]

        recv = server.recieve(recv=np.vstack(sent_measurements),
                              indices=sent_idx)

        server_recieved.append(np.expand_dims(recv, axis=0))

        label = output[sample_idx]
        transmit_dist[label].append(num_transmitted)
        size_dist[label].append(total_bytes)
        total_transmitted += num_transmitted

        if (sample_idx + 1) % PRINT_FREQ == 0:
            print('Completed {0} sequences.'.format(sample_idx + 1), end='\r')

    print()

    recieved = np.vstack(server_recieved)

    # Evaluate the inference model
    predictions = server.predict(inputs=recieved)
    test_accuracy = accuracy_score(y_true=output, y_pred=predictions)
    print('Evaluation Accuracy: {0:.5f}'.format(test_accuracy))

    # Compile some aggregate results for convenience
    total_bytes = sum(sum(byte_list) for byte_list in size_dist.values())
    avg_bytes = total_bytes / num_samples

    print('Avg Bytes Per Sequence: {0:.4f}'.format(avg_bytes))

    avg_transmitted = total_transmitted / num_samples
    print('Avg Messages per Sequence: {0:.4f}/{1} ({2:.4f})'.format(avg_transmitted, seq_length, avg_transmitted / seq_length))

    # Save the result
    result = {
        'accuracy': test_accuracy,
        'avg_bytes': avg_bytes,
        'avg_transmitted': avg_transmitted,
        'byte_dist': size_dist,
        'transmission_dist': transmit_dist,
        'policy': policy_params
    }

    policy_name = '-'.join(t.lower().strip().replace(',', '') for t in str(policy).split())
    policy_file = '{0}_{1}.pkl.gz'.format(policy_name, int(policy.target * 100))
    output_file = os.path.join(output_folder, policy_file)

    save_pickle_gz(result, output_file)

#    # Print out aggregate values in a table format
#    print('Label & Mean (Std) & Count \\\\')
#    for label, transmit_counts in sorted(size_dist.items()):
#        mean = np.average(transmit_counts)
#        std = np.std(transmit_counts)
#        count = len(transmit_counts)
#        print('{0} & {1:.4f} (\\pm {2:.4f}) \\\\'.format(label, mean, std))
#
#    # Print the Avg # Transmitted Across All Samples
#    total_transmitted = sum(sum(counts) for counts in transmit_dist.values())
#    total_elements = inputs.shape[1] * num_samples
#    print('Total Transmitted: {0}/{1}'.format(total_transmitted, total_elements))
#
#    # Run Pairwise t-tests
#    num_labels = len(transmit_dist.keys())
#    pvalue_mat = np.zeros(shape=(num_labels, num_labels))  # [C, C]
#
#    for label_one in sorted(transmit_dist.keys()):
#        for label_two in sorted(transmit_dist.keys()):
#            t_stat, p_val = ttest_ind(a=transmit_dist[label_one],
#                                      b=transmit_dist[label_two],
#                                      equal_var=False)
#
#            pvalue_mat[label_one, label_two] = p_val
#
#    return pvalue_mat
#
#
#def plot_confusion_matrix(pvalue_mat: np.ndarray, should_save: bool):
#    with plt.style.context('seaborn-ticks'):
#        fig, ax = plt.subplots(figsize=(9, 9))
#
#        # Plot the matrix
#        cmap = plt.get_cmap('magma_r')
#        confusion_mat = ax.matshow(pvalue_mat, cmap=cmap)
#
#        # Add data labels
#        for i in range(pvalue_mat.shape[0]):
#            for j in range(pvalue_mat.shape[1]):
#                pval = pvalue_mat[i, j]
#                
#                text = '{0:.2f}'.format(pval) if pval >= 0.01 else '<0.01'
#
#                ax.annotate(text, (i, j), (i - X_OFFSET, j + Y_OFFSET),
#                            bbox=dict(facecolor='white', edgecolor='black'),
#                            fontsize=8)
#
#        # Set axis labels to the class labels
#        ax.set_xticks(np.arange(pvalue_mat.shape[0]))
#        ax.set_yticks(np.arange(pvalue_mat.shape[1]))
#
#        # Set axis labels
#        ax.set_title('P-Values for Transmit Distributions')
#        ax.set_ylabel('Class Label')
#
#        # Create the colorbar
#        fig.colorbar(confusion_mat)
#
#        plt.tight_layout()
#
#        if should_save:
#            plt.savefig('confusion_mat.pdf')
#        else:
#            plt.show()
#


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-folder', type=str, required=True)
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--policy-params', type=str, required=True, nargs='+')
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--max-num-samples', type=int)
    args = parser.parse_args()

    # Unpack the parameter files by detecting directories
    param_files: List[str] = []
    for param_file in args.policy_params:
        if os.path.isdir(param_file):
            file_names = [name for name in os.listdir(param_file) if name.endswith('.json')]
            param_files.extend(os.path.join(param_file, name) for name in file_names)
        else:
            param_files.append(param_file)

    transition_path = os.path.join(args.model_folder, 'transition_model.pkl.gz')
    model_path = os.path.join(args.model_folder, 'model.pkl.gz')
    data_file = os.path.join(args.data_folder, 'test', 'data.h5')

    # Read the input data into memory
    with h5py.File(data_file, 'r') as fin:
        inputs = fin['inputs'][:]
        output = fin['output'][:]

    seq_length = inputs.shape[1]
    num_features = inputs.shape[2]

    for params_path in sorted(param_files):

        print('==========')
        print('Starting {0}'.format(params_path))
        print('==========')

        # Create the server
        server = Server(transition_path=transition_path,
                        inference_path=model_path,
                        seq_length=seq_length)

        # Make the policy
        policy_params = read_json(params_path)
        policy = make_policy(transition_path=transition_path,
                             seq_length=seq_length,
                             num_features=num_features,
                             **policy_params)

        # Execute the simulation
        run(inputs=inputs,
            output=output,
            policy=policy,
            server=server,
            policy_params=policy_params,
            max_num_samples=args.max_num_samples,
            output_folder=args.output_folder)

    # plot_confusion_matrix(pvalue_mat, should_save=False)
