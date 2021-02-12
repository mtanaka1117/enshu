import numpy as np
import sklearn.metrics as metrics
from argparse import ArgumentParser
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from typing import Any, Dict, Tuple

from utils.file_utils import read_pickle_gz


def create_dataset(policy_result: Dict[str, Any], window_size: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:

    bytes_dist = policy_result['byte_dist']

    inputs: List[np.ndarray] = []
    output: List[int] = []

    for label in bytes_dist.keys():
        num_bytes = bytes_dist[label]

        print('Label: {0}, Num Bytes: {1}'.format(label, np.average(num_bytes)))

        for idx in range(0, len(num_bytes) - window_size - stride, stride):
            data_window = num_bytes[idx:idx + window_size]

            inputs.append(np.expand_dims(data_window, axis=0))
            output.append(label)

    return np.vstack(inputs), np.vstack(output).reshape(-1)


def fit_attack_model(inputs: np.ndarray, output: np.ndarray, train_frac: float):

    # Fit the clustering model
    #num_labels = np.amax(output) + 1
    #kmeans = KMeans(n_clusters=num_labels, random_state=4936)

    #cluster_idx = kmeans.fit_predict(inputs)

    ## Create predictions using the Ground-Truth as a 
    ## best-case scenario
    #label_dist: Dict[int, List[int]] = defaultdict(list)

    #for cluster_id, label in zip(cluster_idx, output):
    #    label_dist[cluster_id].append(label)

    #pred_map: Dict[int, int] = dict()
    #for cluster_id, labels in label_dist.items():
    #    label_counts = np.bincount(labels, minlength=num_labels)

    #    print(label_counts)
    #    print(cluster_id)

    #    pred_map[cluster_id] = np.argmax(label_counts)

    #preds: List[int] = []
    #for sample_id in range(inputs.shape[0]):
    #    cluster_id = cluster_idx[sample_id]
    #    preds.append(pred_map[cluster_id])
    #

    #print(metrics.accuracy_score(y_true=output, y_pred=preds))
    
    rand = np.random.RandomState(582)

    # Scale the inputs
    scaler = StandardScaler()
    model_inputs = scaler.fit_transform(inputs)

    # Shuffle the inputs
    sample_idx = np.arange(model_inputs.shape[0])
    rand.shuffle(sample_idx)

    split_point = int(train_frac * model_inputs.shape[0])
    train_idx, test_idx = sample_idx[:split_point], sample_idx[split_point:]

    train_inputs, test_inputs = model_inputs[train_idx], model_inputs[test_idx]
    train_output, test_output = output[train_idx], output[test_idx]

    clf = MLPClassifier(hidden_layer_sizes=[64], alpha=0.1, max_iter=10000, random_state=rand)
    # clf = LogisticRegression(max_iter=5000, random_state=rand)
    clf.fit(train_inputs, train_output)

    train_accuracy = clf.score(train_inputs, train_output)
    test_accuracy = clf.score(test_inputs, test_output)

    most_freq_label = np.bincount(output, minlength=np.amax(output)).argmax()
    most_freq_labels = [most_freq_label for _ in test_output]
    most_freq_acc = metrics.accuracy_score(y_true=test_output, y_pred=most_freq_labels)

    print('Train Accuracy: {0:.5f} ({1})'.format(train_accuracy, len(train_inputs)))
    print('Attack Accuracy: {0:.5f} ({1})'.format(test_accuracy, len(test_inputs)))
    print('Most Freq Accuracy: {0:.5f}'.format(most_freq_acc))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--policy-file', type=str, required=True)
    parser.add_argument('--window-size', type=int, required=True)
    parser.add_argument('--stride', type=int, required=True)
    args = parser.parse_args()

    policy_result = read_pickle_gz(args.policy_file)

    inputs, output = create_dataset(policy_result, window_size=args.window_size, stride=args.stride)

    fit_attack_model(inputs=inputs, output=output, train_frac=0.7)

