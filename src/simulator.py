import numpy as np
import h5py
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import ttest_ind
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Tuple, Dict, Any

from policies import Policy, AdaptivePolicy, RandomPolicy
from utils.file_utils import read_pickle_gz, save_pickle_gz
from utils.data_utils import array_to_fp, array_to_float


X_OFFSET = 0.2
Y_OFFSET = 0.01


def fit_inference_model(data_file: str, output_file: str):
    # Read the input data
    with h5py.File(data_file, 'r') as fin:
        inputs = fin['inputs'][:]  # [N, T, D]
        output = fin['output'][:]  # [N]

    inputs = inputs.reshape(inputs.shape[0], -1)  # [N, T * D]
    
    scaler = StandardScaler().fit(inputs)
    inputs = scaler.transform(inputs)
    
    clf = MLPClassifier(hidden_layer_sizes=[64], alpha=0.01).fit(inputs, output)

    print('Train Accuracy: {0}'.format(clf.score(inputs, output)))
    
    output_dict = {
        'model': clf,
        'scaler': scaler
    }

    save_pickle_gz(output_dict, output_file)


def run(data_file: str, policy: Policy, model: Dict[str, Any], max_num_samples: Optional[int], train_frac: float):
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

            # Scale the estimates
            estimate = policy.get_estimate().reshape(1, -1)  # [1, D]
            estimate = input_scaler.transform(estimate).reshape(-1)

            # Save the estimates for later analysis
            seq_measurements.append(estimate)

        label = output[sample_idx]
        transmit_dist[label].append(num_transmitted)

        # Translate measurements to fixed-point representation
        transmitted_seq, total_bytes = policy.quantize_seq(seq_measurements, num_transmitted=num_transmitted)

        size_dist[label].append(total_bytes)

        measurement_array = transmitted_seq.reshape(1, -1)  # [1, T * D]
        server_measurements.append(measurement_array)

    # Stack inputs
    model_inputs = np.vstack(server_measurements)  # [N, T * D]

    ## Split into training and testing folds
    split_point = int(train_frac * num_samples)
    train_inputs, test_inputs = model_inputs[:split_point], model_inputs[split_point:]
    train_labels, test_labels = output[:split_point], output[split_point:]

    ## Normalize the inputs
    #scaler = StandardScaler().fit(train_inputs)
    #train_inputs = scaler.transform(train_inputs)
    #test_inputs = scaler.transform(test_inputs)

    # Fit the inference model
    # clf = MLPClassifier(hidden_layer_sizes=[64], alpha=0.01, random_state=281, max_iter=500)
    # clf.fit(train_inputs, train_labels)

    # Evaluate the inference model
    print('Train Accuracy: {0:.5f}'.format(clf.score(train_inputs, train_labels)))
    print('Test Accuracy: {0:.5f}'.format(clf.score(test_inputs, test_labels)))

    # Print out aggregate values in a table format
    print('Label & Mean (Std) & Count \\\\')
    for label, transmit_counts in sorted(size_dist.items()):
        mean = np.average(transmit_counts)
        std = np.std(transmit_counts)
        count = len(transmit_counts)
        print('{0} & {1:.4f} (\\pm {2:.4f}) \\\\'.format(label, mean, std))

    # Print the Avg # Transmitted Across All Samples
    total_transmitted = sum(sum(counts) for counts in transmit_dist.values())
    total_elements = inputs.shape[1] * num_samples
    print('Total Transmitted: {0}/{1}'.format(total_transmitted, total_elements))

    # Run Pairwise t-tests
    num_labels = len(transmit_dist.keys())
    pvalue_mat = np.zeros(shape=(num_labels, num_labels))  # [C, C]

    for label_one in sorted(transmit_dist.keys()):
        for label_two in sorted(transmit_dist.keys()):
            t_stat, p_val = ttest_ind(a=transmit_dist[label_one],
                                      b=transmit_dist[label_two],
                                      equal_var=False)

            pvalue_mat[label_one, label_two] = p_val

    return pvalue_mat


def plot_confusion_matrix(pvalue_mat: np.ndarray, should_save: bool):
    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots(figsize=(9, 9))

        # Plot the matrix
        cmap = plt.get_cmap('magma_r')
        confusion_mat = ax.matshow(pvalue_mat, cmap=cmap)

        # Add data labels
        for i in range(pvalue_mat.shape[0]):
            for j in range(pvalue_mat.shape[1]):
                pval = pvalue_mat[i, j]
                
                text = '{0:.2f}'.format(pval) if pval >= 0.01 else '<0.01'

                ax.annotate(text, (i, j), (i - X_OFFSET, j + Y_OFFSET),
                            bbox=dict(facecolor='white', edgecolor='black'),
                            fontsize=8)

        # Set axis labels to the class labels
        ax.set_xticks(np.arange(pvalue_mat.shape[0]))
        ax.set_yticks(np.arange(pvalue_mat.shape[1]))

        # Set axis labels
        ax.set_title('P-Values for Transmit Distributions')
        ax.set_ylabel('Class Label')

        # Create the colorbar
        fig.colorbar(confusion_mat)

        plt.tight_layout()

        if should_save:
            plt.savefig('confusion_mat.pdf')
        else:
            plt.show()


# Fit the inference model
#data_file = 'datasets/uci_har/validation/data.h5'
#fit_inference_model(data_file, output_file='inference_model.pkl.gz')


transition_path = 'transition_model.pkl.gz'

#policy = AdaptivePolicy(transition_path=transition_path,
#                        threshold=0.18,
#                        target=0.7,
#                        precision=6,
#                        width=8)


policy = RandomPolicy(transition_path=transition_path,
                      target=0.7,
                      precision=6,
                      width=8)

inference_model = read_pickle_gz('saved_models/uci_har/model.pkl.gz')

pvalue_mat = run(data_file='datasets/uci_har/test/data.h5',
                 policy=policy,
                 model=inference_model,
                 max_num_samples=None,
                 train_frac=0.7)

plot_confusion_matrix(pvalue_mat, should_save=False)
