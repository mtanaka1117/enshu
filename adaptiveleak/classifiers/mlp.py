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

    def build(self, input_shape: Tuple[int, int], num_classes: int) -> keras.models.Model:
        # Create the input layer
        input_layer = keras.layers.Input(input_shape)

        # Stack all features into a single vector and apply dropout
        flattened_input = keras.layers.Flatten()(input_layer)
        flattened_input = keras.layers.Dropout(0.1)(flattened_input)

        # Apply the hidden layers
        hidden_1 = keras.layers.Dense(self.hidden_units, activation='relu')(flattened_input)
        hidden_1 = keras.layers.Dropout(0.2)(hidden_1)

        hidden_2 = keras.layers.Dense(self.hidden_units, activation='relu')(hidden_1)
        hidden_2 = keras.layers.Dropout(0.2)(hidden_2)

        hidden_3 = keras.layers.Dense(self.hidden_units, activation='relu')(hidden_2)
        hidden_3 = keras.layers.Dropout(0.3)(hidden_3)

        # Apply the output layer
        probs = keras.layers.Dense(num_classes, activation='softmax')(hidden_3)

        # Create and Return the model
        model = keras.models.Model(inputs=input_layer, outputs=probs)

        return model
