import numpy as np
import tensorflow.keras as keras
from typing import Tuple

from .dnn import NeuralNetwork


class MLP(NeuralNetwork):

    def __init__(self, batch_size: int, hidden_units: int):
        super().__init__(batch_size=batch_size)
        self._hidden_units = hidden_units

    @property
    def name(self) -> str:
        return 'mlp'

    @property
    def lr_decay(self) -> int:
        return 200

    @property
    def hidden_units(self) -> int:
        return self._hidden_units

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
