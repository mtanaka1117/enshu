import numpy as np
import h5py
import os.path
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
from scipy.stats import ttest_ind
from typing import Optional, List, Tuple, Dict, Any

from policies import Policy, make_policy
from utils.file_utils import read_pickle_gz, save_pickle_gz, read_json
from utils.data_utils import array_to_fp, array_to_float


X_OFFSET = 0.2
Y_OFFSET = 0.01


def run(data_file: str, policy: Policy, model: Dict[str, Any], should_quantize: bool, max_num_samples: Optional[int], output_file: str):
    # Read the input data
    with h5py.File(data_file, 'r') as fin:
        inputs = fin['inputs'][:]
        output = fin['output'][:]

    # Get the number of samples to evaluate
    num_samples = inputs.shape[0] if max_num_samples is None else max_num_samples
    inputs = inputs[:num_samples]  # [N, T, D]
    output = output[:num_samples]  # [N]

    # Count the number of transmitted values per class
    transmit_dist: DefaultDict[int, List[int]] = defaultdict(list)
    size_dist: DefaultDict[int, List[int]] = defaultdict(list)

    # Unpack the already-trained model
    input_scaler = model['scaler']
    clf = model['model']

    # Store the measurements as observed on the server
    server_measurements: List[np.ndarray] = []

    for sample_idx in range(num_samples):
        seq_features = inputs[sample_idx]  # [T, D]

        seq_measurements: List[np.ndarray] = []

        # Reset the policy before every sequence
        policy.reset()

        num_transmitted = 0
        for input_features in seq_features:
            # Update the estimate
            policy.transition()

            # Determine whether to transmit the measurement
            measurement = np.expand_dims(input_features, axis=-1)  # [D, 1]
            num_transmitted += policy.transmit(measurement=measurement)

            # Save the estimates for later analysis
            seq_measurements.append(policy.get_estimate().reshape(-1))

        label = output[sample_idx]
        transmit_dist[label].append(num_transmitted)

        # Scale the measurements
        scaled_seq = input_scaler.transform(seq_measurements)

        # Translate measurements to fixed-point representation (and back)
        if should_quantize:
            transmitted_seq, total_bytes = policy.quantize_seq(scaled_seq,
                                                               num_transmitted=num_transmitted)
        else:
            transmitted_seq = scaled_seq

            num_features = len(transmitted_seq[0])
            total_bits = policy.width * num_transmitted * num_features
            total_bytes = total_bits / 8

        size_dist[label].append(total_bytes)

        measurement_array = transmitted_seq.reshape(1, -1)  # [1, T * D]
        server_measurements.append(measurement_array)

    # Stack inputs
    model_inputs = np.vstack(server_measurements)  # [N, T * D]

    # Evaluate the inference model
    test_accuracy = clf.score(model_inputs, output)
    print('Evaluation Accuracy: {0:.5f}'.format(test_accuracy))

    # Compile some aggregate results for convenience
    total_bytes = sum(sum(byte_list) for byte_list in size_dist.values())
    avg_bytes = total_bytes / num_samples

    print('Average Bytes Per Sequence: {0:.4f}'.format(avg_bytes))

    # Save the result
    result = {
        'accuracy': test_accuracy,
        'avg_bytes': avg_bytes,
        'byte_dist': size_dist,
        'is_quantized': should_quantize,
        'transmission_dist': transmit_dist,
        'policy': policy
    }

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
    parser.add_argument('--policy-params', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--should-quantize', action='store_true')
    args = parser.parse_args()

    transition_path = os.path.join(args.model_folder, 'transition_model.pkl.gz')
    model_path = os.path.join(args.model_folder, 'model.pkl.gz')
    data_file = os.path.join(args.data_folder, 'test', 'data.h5')

    # Make the policy
    policy_params = read_json(args.policy_params)
    policy = make_policy(transition_path=transition_path, **policy_params)

    # Read the inference model
    inference_model = read_pickle_gz(model_path)

    pvalue_mat = run(data_file=data_file,
                     policy=policy,
                     model=inference_model,
                     max_num_samples=None,
                     output_file=args.output_file,
                     should_quantize=args.should_quantize)

    # plot_confusion_matrix(pvalue_mat, should_save=False)
