import numpy as np
import os
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ndcg_score, top_k_accuracy_score, dcg_score, confusion_matrix
from typing import Any, Dict, Tuple, List, DefaultDict

from adaptiveleak.utils.analysis import geometric_mean
from adaptiveleak.utils.constants import ENCODING, POLICIES
from adaptiveleak.utils.file_utils import read_json_gz, save_json_gz, iterate_dir, make_dir


AttackResult = namedtuple('AttackResult', ['accuracy', 'precision', 'recall', 'f1', 'ndcg', 'dcg', 'top2', 'confusion_mat'])


class AttackResultList:

    def __init__(self):
        self.accuracy: List[float] = []
        self.precision: List[float] = []
        self.recall: List[float] = []
        self.f1: List[float] = []
        self.ndcg: List[float] = []
        self.dcg: List[float] = []
        self.top2: List[float] = []
        self.confusion_mat: List[np.ndarray] = []

    def append(self, result: AttackResult):
        self.accuracy.append(result.accuracy)
        self.precision.append(result.precision)
        self.recall.append(result.recall)
        self.f1.append(result.f1)
        self.ndcg.append(result.ndcg)
        self.dcg.append(result.dcg)
        self.top2.append(result.top2)
        self.confusion_mat.append(result.confusion_mat.astype(int))

    def as_dict(self) -> Dict[str, List[float]]:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'ndcg': self.ndcg,
            'dcg': self.dcg,
            'top2': self.top2,
            'confusion_mat': [mat.tolist() for mat in self.confusion_mat]
        }


def get_stats(true: np.ndarray, pred_probs: np.ndarray, num_labels: int) -> AttackResult:
    """
    Returns the statistics comparing the true and predicted values.
    """
    assert len(true.shape) == 1, 'Must provide a 1d array of true labels'
    assert len(pred_probs.shape) == 2, 'Must provide a 2d array of predicted probabilities'

    # Compute the NDCG using a one-hot relevance encoding
    true_relevance = np.zeros_like(pred_probs)

    for sample_idx, label in enumerate(true):
        true_relevance[sample_idx, label] = 1

    ndcg = ndcg_score(y_true=true_relevance,
                      y_score=pred_probs)

    dcg = dcg_score(y_true=true_relevance,
                    y_score=pred_probs)

    num_classes = np.amax(true) + 1
    pred = np.argmax(pred_probs, axis=-1)

    if num_classes > 2:
        top2 = top_k_accuracy_score(y_true=true,
                                    y_score=pred_probs,
                                    k=2)
    else:
        top2 = 1.0

    labels = list(range(num_labels))

    return AttackResult(accuracy=accuracy_score(y_true=true, y_pred=pred),
                        precision=precision_score(y_true=true, y_pred=pred, average='micro', zero_division=0),
                        recall=recall_score(y_true=true, y_pred=pred, average='micro'),
                        f1=f1_score(y_true=true, y_pred=pred, average='macro'),
                        ndcg=ndcg,
                        dcg=dcg,
                        top2=top2,
                        confusion_mat=confusion_matrix(y_true=true, y_pred=pred, labels=labels))


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
    assert len(message_sizes) == len(labels), 'Must provide the same number of message sizes and labels'

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
    num_labels = np.amax(labels) + 1

    train_results = AttackResultList()
    test_results = AttackResultList()
    most_freq_results = AttackResultList()

    kf = KFold(n_splits=5, random_state=rand, shuffle=True)
    num_train_samples = int(0.8 * num_samples)
    num_test_samples = num_samples - num_train_samples


    for train_idx, test_idx in kf.split(message_sizes):
        # Create the input features based on the given train-test splits. This methodology
        # avoids a case where the training and testing sets have information from the same
        # (size, label) pair.
        train_inputs, train_labels = create_dataset(message_sizes=message_sizes[train_idx],
                                                    labels=labels[train_idx],
                                                    window_size=window_size,
                                                    num_samples=num_train_samples,
                                                    rand=rand)

        test_inputs, test_labels = create_dataset(message_sizes=message_sizes[test_idx],
                                                  labels=labels[test_idx],
                                                  window_size=window_size,
                                                  num_samples=num_train_samples,
                                                  rand=rand)

        # Create the fit the model
        clf = AdaBoostClassifier(n_estimators=50)
        clf.fit(train_inputs, train_labels)

        # Get the training and testing predictions
        train_probs = clf.predict_proba(train_inputs)
        test_probs = clf.predict_proba(test_inputs)

        train_stats = get_stats(true=train_labels, pred_probs=train_probs, num_labels=num_labels)
        test_stats = get_stats(true=test_labels, pred_probs=test_probs, num_labels=num_labels)

        # Get the most frequent accuracy based on the training set
        most_freq_label_counts = np.bincount(train_labels, minlength=num_labels)
        most_freq_label = np.argmax(most_freq_label_counts)

        most_freq_probs = np.zeros_like(test_probs)
        most_freq_probs[:, most_freq_label] = 1

        most_freq_stats = get_stats(true=test_labels, pred_probs=most_freq_probs, num_labels=num_labels)

        train_results.append(train_stats)
        test_results.append(test_stats)
        most_freq_results.append(most_freq_stats)

    print('Train Accuracy: {0:.5f} ({1:.5f})'.format(np.average(train_results.accuracy), np.std(train_results.accuracy)))
    print('Test Accuracy: {0:.5f} ({1:.5f})'.format(np.average(test_results.accuracy), np.std(test_results.accuracy)))
    print('Most Freq Accuracy: {0:.5f} ({1:.5f})'.format(np.average(most_freq_results.accuracy), np.std(test_results.accuracy)))

    return dict(train=train_results.as_dict(),
                test=test_results.as_dict(),
                most_freq=most_freq_results.as_dict(),
                count=num_samples)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--policy', type=str, required=True, choices=POLICIES)
    parser.add_argument('--encoding', type=str, required=True, choices=ENCODING)
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
