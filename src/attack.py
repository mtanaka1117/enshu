import numpy as np
import sklearn.metrics as metrics
import os
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from typing import Any, Dict, Tuple, List

from utils.file_utils import read_pickle_gz, save_pickle_gz


AttackResult = namedtuple('AttackResult', ['train_accuracy', 'num_train', 'test_accuracy', 'num_test', 'most_freq_accuracy'])


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
    clf.fit(train_inputs, train_output)

    train_accuracy = clf.score(train_inputs, train_output)
    test_accuracy = clf.score(test_inputs, test_output)

    most_freq_label = np.bincount(output, minlength=np.amax(output)).argmax()
    most_freq_labels = [most_freq_label for _ in test_output]
    most_freq_acc = metrics.accuracy_score(y_true=test_output, y_pred=most_freq_labels)

    print('Train Accuracy: {0:.5f} ({1})'.format(train_accuracy, len(train_inputs)))
    print('Attack Accuracy: {0:.5f} ({1})'.format(test_accuracy, len(test_inputs)))
    print('Most Freq Accuracy: {0:.5f}'.format(most_freq_acc))

    return AttackResult(train_accuracy=train_accuracy,
                        num_train=len(train_inputs),
                        test_accuracy=test_accuracy,
                        num_test=len(test_inputs),
                        most_freq_accuracy=most_freq_acc)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--policy-files', type=str, required=True, nargs='+')
    parser.add_argument('--window-size', type=int, required=True)
    parser.add_argument('--stride', type=int, required=True)
    args = parser.parse_args()

    policy_files: List[str] = []
    for policy_file in args.policy_files:

        if os.path.isdir(policy_file):
            file_names = [name for name in os.listdir(policy_file) if name.endswith('.pkl.gz')]
            policy_files.extend((os.path.join(policy_file, name) for name in file_names))
        else:
            policy_files.append(policy_file)

    for policy_file in sorted(policy_files):
        print('==========')
        print('Starting {0}'.format(policy_file))
        print('==========')
        
        policy_result = read_pickle_gz(policy_file)

        inputs, output = create_dataset(policy_result, window_size=args.window_size, stride=args.stride)

        attack_result = fit_attack_model(inputs=inputs, output=output, train_frac=0.7)

        # Save the attack result
        policy_result['attack'] = attack_result._asdict()
        save_pickle_gz(policy_result, policy_file)
