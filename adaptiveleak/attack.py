import numpy as np
import sklearn.metrics as metrics
import os
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from typing import Any, Dict, Tuple, List

from adaptiveleak.classifiers.mlp import MLP
from adaptiveleak.utils.file_utils import read_json_gz, save_json_gz


AttackResult = namedtuple('AttackResult', ['train_accuracy', 'num_train', 'test_accuracy', 'num_test', 'most_freq_accuracy'])


def create_dataset(message_sizes: List[int], labels: List[int], window_size: int, num_samples: int, rand: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates the attack dataset by randomly sampling message sizes of the given window.

    Args:
        message_sizes: The size of each message (in bytes)
        labels: The true label for each message
        window_size: The size of the model's features (D)
        num_samples: The number of samples to create
        rand: The random state used to create samples in a reproducible manner
    Returns:
        A tuple of two elements.
            (1) A [N, D] array of input features composed of message sizes
            (2) A [N] array of labels for each input
    """
    num_messages = len(message_sizes)

    # Group the message sizes by label
    bytes_dist: DefaultDict[int, List[int]] = defaultdict(list)

    for label, size in zip(labels, message_sizes):
        bytes_dist[label].append(size)

    inputs: List[np.ndarray] = []
    output: List[int] = []

    for label in bytes_dist.keys():
        sizes = bytes_dist[label]

        # print('Label: {0}, Num Bytes: {1} (Std: {2}, Count: {3})'.format(label, np.average(sizes), np.std(sizes), len(sizes)))

        num_to_create = int(round(num_samples * (len(sizes) / num_messages)))

        for _ in range(num_to_create):
            raw_sizes = rand.choice(sizes, size=window_size)  # [D]
            features = np.array([np.average(raw_sizes), np.median(raw_sizes), np.std(raw_sizes), np.max(raw_sizes), np.min(raw_sizes)])

            inputs.append(np.expand_dims(features, axis=0))
            output.append(label)

    return np.vstack(inputs), np.vstack(output).reshape(-1)


def fit_attack_model(message_sizes: List[int], labels: List[int], window_size: int, num_samples: int, train_frac: float, val_frac: float):
    """
    Fits the attacker model which predicts labels from message sizes.

    Args:
        inputs: A [
    """
    assert len(message_sizes) == len(labels), 'Must provide the same number of messages ({0}) as labels ({1}).'.format(len(message_sizes), len(labels))

    # Shuffle the data together
    rand = np.random.RandomState(582)
    
    zipped = list(zip(message_sizes, labels))
    rand.shuffle(zipped)
    message_sizes, labels = zip(*zipped)

    # Split the data
    train_split = int(train_frac * len(message_sizes))
    val_split = int((train_frac + val_frac) * len(message_sizes))

    train_sizes = message_sizes[:train_split]
    val_sizes = message_sizes[train_split:val_split]
    test_sizes = message_sizes[val_split:]
    
    train_labels = labels[:train_split]
    val_labels = labels[train_split:val_split]
    test_labels = labels[val_split:]

    # Create the data-sets
    num_train = int(train_frac * num_samples)
    num_val = int(val_frac * num_samples)
    num_test = num_samples - num_train - num_val

    train_inputs, train_outputs = create_dataset(message_sizes=train_sizes,
                                                 labels=train_labels,
                                                 window_size=window_size,
                                                 num_samples=num_train,
                                                 rand=rand)

    val_inputs, val_outputs = create_dataset(message_sizes=val_sizes,
                                             labels=val_labels,
                                             window_size=window_size,
                                             num_samples=num_val,
                                             rand=rand)

    test_inputs, test_outputs = create_dataset(message_sizes=test_sizes,
                                               labels=test_labels,
                                               window_size=window_size,
                                               num_samples=num_test,
                                               rand=rand)

    # Scale the inputs
    scaler = StandardScaler()
    train_inputs = scaler.fit_transform(train_inputs)
    val_inputs = scaler.transform(val_inputs)
    test_inputs = scaler.transform(test_inputs)

    clf = MLP(batch_size=16, hidden_units=64)
    # clf = MLPClassifier(hidden_layer_sizes=[32], alpha=0.1, max_iter=10000, random_state=rand)
    
    most_freq_label = np.bincount(train_outputs, minlength=np.amax(train_outputs)).argmax()
    most_freq_labels = [most_freq_label for _ in test_outputs]
    most_freq_acc = metrics.accuracy_score(y_true=test_outputs, y_pred=most_freq_labels)

    print('Most Freq Accuracy: {0:.5f}'.format(most_freq_acc))

    clf.fit(train_inputs=train_inputs,
            train_labels=train_outputs,
            val_inputs=val_inputs,
            val_labels=val_outputs,
            num_epochs=1,
            save_folder='attack_models')

    # Load the best model
    clf.restore('attack_models')

    train_accuracy = clf.accuracy(train_inputs, train_outputs)
    val_accuracy = clf.accuracy(val_inputs, val_outputs)
    test_accuracy = clf.accuracy(test_inputs, test_outputs)

    print('Train Accuracy: {0:.5f} ({1})'.format(train_accuracy, len(train_inputs)))
    print('Val Accuracy: {0:.5f} ({1})'.format(val_accuracy, len(val_inputs)))
    print('Attack Accuracy: {0:.5f} ({1})'.format(test_accuracy, len(test_inputs)))
    print('Most Freq Accuracy: {0:.5f}'.format(most_freq_acc))

    return dict(train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                test_accuracy=test_accuracy,
                num_train=len(train_inputs),
                num_val=len(val_inputs),
                num_test=len(test_inputs),
                most_freq_accuracy=most_freq_acc)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--policy-files', type=str, required=True, nargs='+')
    parser.add_argument('--window-size', type=int, required=True)
    parser.add_argument('--num-samples', type=int, required=True)
    args = parser.parse_args()

    policy_files: List[str] = []
    for policy_file in args.policy_files:

        if os.path.isdir(policy_file):
            file_names = [name for name in os.listdir(policy_file) if name.endswith('.json.gz')]
            policy_files.extend((os.path.join(policy_file, name) for name in file_names))
        else:
            policy_files.append(policy_file)

    for policy_file in sorted(policy_files):
        print('==========')
        print('Starting {0}'.format(policy_file))
        print('==========')
        
        policy_result = read_json_gz(policy_file)

        attack_result = fit_attack_model(message_sizes=policy_result['num_bytes'],
                                        labels=policy_result['labels'],
                                        window_size=args.window_size,
                                        train_frac=0.7,
                                        val_frac=0.15,
                                        num_samples=args.num_samples)

        # Save the attack result
        policy_result['attack'] = attack_result
        save_json_gz(policy_result, policy_file)
