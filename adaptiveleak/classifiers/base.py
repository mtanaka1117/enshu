import numpy as np
from sklearn.metrics import accuracy_score


class BaseClassifier:

    def __init__(self):
        self._rand = np.random.RandomState(seed=4925)

    @property
    def name(self) -> str:
        raise NotImplementedError()

    def restore(self, save_folder: str):
        """
        Restores the given model
        """
        raise NotImplementedError()

    def fit(self, train_inputs: np.ndarray, train_labels: np.ndarray, val_inputs: np.ndarray, val_labels: np.ndarray, num_epochs: int, save_folder: str):
        """
        Fits the given classifier

        Args:
            train_inputs: A [N, D] array of input features
            train_labels: A [N] array of labels
            val_inputs: A [M, D] array of validation input features
            val_labels: A [M] array of validation labels
            save_folder: The folder in which to place the results
            num_epochs: The number of training epochs
        """
        raise NotImplementedError()

    def predict_probs(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predicts the probabilities on the given inputs.

        Args:
            inputs: A [N, D] array of input features
        Returns:
            A [N, K] array of class probabilities for each sample
        """
        raise NotImplementedError()

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predicts the labels for each classifier

        """
        probs = self.predict_probs(inputs=inputs)  # [N, K]
        return np.argmax(probs, axis=-1)

    def accuracy(self, inputs: np.ndarray, labels: np.ndarray) -> float:
        """
        Computes the accuracy of the given model on the provided data.

        Args:
            inputs: A [N, D] array of input features
            labels: A [N] array of output features
        Returns:
            The accuracy on the given dataset
        """
        assert len(labels.shape) == 1, 'Labels must be a 1d array'
        predictions = self.predict(inputs)  # [N]
        return accuracy_score(y_pred=predictions, y_true=labels)
