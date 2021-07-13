import numpy as np
import os
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Any, Dict, Tuple, List, DefaultDict

from adaptiveleak.utils.analysis import geometric_mean
from adaptiveleak.utils.file_utils import read_json_gz, save_json_gz, iterate_dir, make_dir


AttackResult = namedtuple('AttackResult', ['accuracy', 'precision', 'recall', 'f1'])


class AttackResultList:

    def __init__(self):
        self.accuracy: List[float] = []
        self.precision: List[float] = []
        self.recall: List[float] = []
        self.f1: List[float] = []

    def append(self, result: AttackResult):
        self.accuracy.append(result.accuracy)
        self.precision.append(result.precision)
        self.recall.append(result.recall)
        self.f1.append(result.f1)

    def as_dict(self) -> Dict[str, List[float]]:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1
        }


def get_stats(true: np.ndarray, pred: np.ndarray) -> AttackResult:
    """
    Returns the statistics comparing the true and predicted values.
    """
    assert len(true.shape) == 1, 'Must provide a 1d array of true labels'
    assert len(pred.shape) == 1, 'Must provide a 1d array of predicted labels'

    return AttackResult(accuracy=accuracy_score(y_true=true, y_pred=pred),
                        precision=precision_score(y_true=true, y_pred=pred, average='macro', zero_division=0),
                        recall=recall_score(y_true=true, y_pred=pred, average='macro'),
                        f1=f1_score(y_true=true, y_pred=pred, average='macro'))


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

        num_to_create = int(round(num_samples * (len(sizes) / num_messages)))

        for _ in range(num_to_create):
            raw_sizes = rand.choice(sizes, size=window_size)  # [D]

            iqr = np.percentile(raw_sizes, 75) - np.percentile(raw_sizes, 25)
            features = [np.average(raw_sizes), np.std(raw_sizes), np.median(raw_sizes), np.max(raw_sizes), np.min(raw_sizes), iqr, geometric_mean(raw_sizes)]

            inputs.append(np.expand_dims(features, axis=0))
            output.append(label)

    return np.vstack(inputs), np.vstack(output).reshape(-1)


def fit_attack_model(message_sizes: np.array, labels: np.array, window_size: int, num_samples: int, name: str, save_folder: str):
    """
    Fits the attacker model which predicts labels from message sizes.
    """
    assert len(message_sizes) == len(labels), 'Must provide the same number of messages ({0}) as labels ({1}).'.format(len(message_sizes), len(labels))

    rand = np.random.RandomState(582)

    # Create the data-set
    inputs, outputs = create_dataset(message_sizes=message_sizes,
                                     labels=labels,
                                     window_size=window_size,
                                     num_samples=num_samples,
                                     rand=rand)

    train_results = AttackResultList()
    test_results = AttackResultList()
    most_freq_results = AttackResultList()

    kf = KFold(n_splits=5, random_state=rand, shuffle=True)

    for train_idx, test_idx in kf.split(inputs):
        # Split the data
        train_inputs, test_inputs = inputs[train_idx], inputs[test_idx]
        train_labels, test_labels = outputs[train_idx], outputs[test_idx]

        # Create the fit the model
        clf = AdaBoostClassifier(n_estimators=50)
        clf.fit(train_inputs, train_labels)

        # Get the training and testing predictions
        train_pred = clf.predict(train_inputs)
        test_pred = clf.predict(test_inputs)

        train_stats = get_stats(true=train_labels, pred=train_pred)
        test_stats = get_stats(true=test_labels, pred=test_pred)

        train_results.append(train_stats)
        test_results.append(test_stats)

        # Get the most frequent accuracy based on the training set
        most_freq_label = np.bincount(train_labels, minlength=np.amax(train_labels)).argmax()
        most_freq_labels = np.array([most_freq_label for _ in test_labels])
        most_freq_stats = get_stats(true=test_labels, pred=most_freq_labels)

        train_results.append(train_stats)
        test_results.append(test_stats)
        most_freq_results.append(most_freq_stats)

    print('Train Accuracy: {0:.5f}'.format(np.average(train_results.accuracy)))
    print('Test Accuracy: {0:.5f}'.format(np.average(test_results.accuracy)))
    print('Most Freq Accuracy: {0:.5f}'.format(np.average(most_freq_results.accuracy)))

    return dict(train=train_results.as_dict(),
                test=test_results.as_dict(),
                most_freq=most_freq_results.as_dict(),
                count=len(inputs))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--encoding', type=str, required=True, choices=['standard', 'group'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--date', type=str, required=True)
    parser.add_argument('--window-size', type=int, required=True)
    parser.add_argument('--num-samples', type=int, required=True)
    args = parser.parse_args()

    policy_name = '{0}_{1}'.format(args.policy, args.encoding)
    policy_folder = os.path.join('..', 'saved_models', args.dataset, args.date, policy_name)

    save_folder = os.path.join(policy_folder, 'attack_models')
    make_dir(save_folder)

    for path in iterate_dir(policy_folder, '.*json.gz'):
    
        print('===== STARTING {0} ====='.format(path))

        policy_result = read_json_gz(path)
        
        model_name = os.path.basename(path)
        model_name = model_name.split('.')[0]

        attack_result = fit_attack_model(message_sizes=np.array(policy_result['num_bytes']),
                                         labels=np.array(policy_result['labels']),
                                         window_size=args.window_size,
                                         num_samples=args.num_samples,
                                         name=model_name,
                                         save_folder=save_folder)

        # Save the attack result
        policy_result['attack'] = attack_result
        save_json_gz(policy_result, path)
