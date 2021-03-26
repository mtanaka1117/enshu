import tensorflow.keras as keras
import numpy as np
import os.path
from typing import Any, Tuple
from sklearn.model_selection import train_test_split

from .base import BaseClassifier


class NeuralNetwork(BaseClassifier):

    def __init__(self, batch_size: int, train_frac: float, learning_rate: float):
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._train_frac = train_frac

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def train_frac(self) -> float:
        return self._train_frac

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def weights_file_name(self) -> str:
        return '{0}_best.hd5'.format(self.name)

    @property
    def metadata_file_name(self) -> str:
        return '{0}_metadata.pkl.gz'.format(self.name)

    def build(self, input_shape: Tuple[int, int], num_classes: int) -> Tuple[keras.models.Model, Any]:
        raise NotImplementedError()

    def restore(self, save_folder: str):
        """
        Restores the given model
        """
        weights_path = os.path.join(save_folder, self.weights_file_name)
        self._model = keras.models.load_model(weights_path)

    def fit(self, train_inputs: np.ndarray, train_labels: np.ndarray, val_inputs: np.ndarray, val_labels: np.ndarray, num_epochs: int, save_folder: str):
        """
        Fits the given classifier

        Args:
            train_inputs: A [N, T, D] array of training input features
            train_labels: A [N] array of training labels
            val_inputs: A [M, T, D] array of validation input features
            val_labels: A [M] array of validation labels
            num_epochs: The number of training epochs
            save_folder: The folder in which to place the results
        """
        assert len(train_inputs.shape) == 3, 'Must provide a 3d training input'
        assert len(val_inputs.shape) == 3, 'Must provide a 3d validation input'

        # Build the model
        input_shape = train_inputs.shape[1:]
        num_classes = np.max(train_labels) + 1

        model = self.build(input_shape=input_shape, num_classes=num_classes)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      metrics=['accuracy'])
        model.summary()

        # Set Training Callbacks
        lr_decay = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.5,
                                                     patience=100,
                                                     min_lr=1e-5)

        save_path = os.path.join(save_folder, self.weights_file_name)
        checkpoint = keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)

        callbacks = [lr_decay, checkpoint]

        # Train the model
        batch_size = min(train_inputs.shape[0], self.batch_size)
        history = model.fit(train_inputs, train_labels,
                            batch_size=batch_size,
                            epochs=num_epochs,
                            verbose=True,
                            validation_data=(val_inputs, val_labels),
                            callbacks=callbacks)

        self._model = model

        # TODO: Save the training results

    def predict_probs(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predicts the probabilities on the given inputs.

        Args:
            inputs: A [N, T, D] array of input features
        Returns:
            A [N, K] array of class probabilities for each sample
        """
        return self._model.predict(inputs)
