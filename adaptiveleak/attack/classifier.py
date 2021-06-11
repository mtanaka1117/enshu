import numpy as np
import os
import tensorflow.keras as keras

from sklearn.metrics import accuracy_score
from typing import Any, List


class AttackClassifier:

    def __init__(self, name: str):
        self._rand = np.random.RandomState(seed=3146)
        self._optimizer = keras.optimizers.Adam()
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def batch_size(self) -> int:
        return 16

    @property
    def lr_decay(self) -> int:
        return 200

    @property
    def hidden_units(self) -> int:
        return 200

    @property
    def min_learning_rate(self) -> float:
        return 1e-4

    @property
    def weights_file_name(self) -> str:
        return '{0}_best.hd5'.format(self.name)

    @property
    def metadata_file_name(self) -> str:
        return '{0}_metadata.pkl.gz'.format(self.name)


    def build(self, num_features: int, num_classes: int) -> keras.models.Model:
        # Create the input layer
        inputs = keras.layers.Input(shape=(num_features, ))

        # Apply dropout to inputs
        inputs_with_dropout = keras.layers.Dropout(0.1)(inputs)

        # Apply the hidden layers
        hidden_1 = keras.layers.Dense(self.hidden_units, activation='relu')(inputs_with_dropout)
        hidden_1 = keras.layers.Dropout(0.2)(hidden_1)

        hidden_2 = keras.layers.Dense(self.hidden_units, activation='relu')(hidden_1)
        hidden_2 = keras.layers.Dropout(0.2)(hidden_2)

        hidden_3 = keras.layers.Dense(self.hidden_units, activation='relu')(hidden_2)
        hidden_3 = keras.layers.Dropout(0.3)(hidden_3)

        # Apply the output layer
        probs = keras.layers.Dense(num_classes, activation='softmax')(hidden_3)

        # Create and Return the model
        model = keras.models.Model(inputs=inputs, outputs=probs)

        return model

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
            train_inputs: A [N, T, D] array of input features
            train_labels: A [N] array of labels
            val_inputs: A [M, T, D] array of validation input features
            val_labels: A [M] array of validation labels
            save_folder: The folder in which to place the results
            num_epochs: The number of training epochs
        """
        assert len(train_inputs.shape) == 2, 'Must provide a 3d training input'
        assert len(val_inputs.shape) == 2, 'Must provide a 3d validation input'

        # Build the model
        num_features = train_inputs.shape[-1]
        num_classes = np.max(train_labels) + 1

        model = self.build(num_features=num_features,
                           num_classes=num_classes)

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=self._optimizer,
                      metrics=['accuracy'])
        model.summary()

        # Set Training Callbacks
        lr_decay = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.5,
                                                     patience=self.lr_decay,
                                                     min_lr=self.min_learning_rate)

        save_path = os.path.join(save_folder, self.weights_file_name)
        checkpoint = keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)

        callbacks = [lr_decay, checkpoint]

        # Train the model
        batch_size = min(int(train_inputs.shape[0] / 10), self.batch_size)
        history = model.fit(train_inputs, train_labels,
                            batch_size=batch_size,
                            epochs=num_epochs,
                            verbose=True,
                            validation_data=(val_inputs, val_labels),
                            callbacks=callbacks)

        self._model = model

    def predict_probs(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predicts the probabilities on the given inputs.

        Args:
            inputs: A [N, D] array of input features
        Returns:
            A [N, K] array of class probabilities for each sample
        """
        return self._model.predict(inputs)

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
