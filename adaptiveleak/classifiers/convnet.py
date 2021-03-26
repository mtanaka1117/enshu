import numpy as np
import tensorflow.keras as keras
import os.path
from typing import Tuple

from .dnn import NeuralNetwork


class ConvNet(NeuralNetwork):
    
    def __init__(self, batch_size: int, train_frac: float, learning_rate: float, num_filters: int):
        super().__init__(batch_size=batch_size,
                         train_frac=train_frac,
                         learning_rate=learning_rate)
        self._num_filters = num_filters
 
    @property
    def name(self) -> str:
        return 'convnet'

    @property
    def num_filters(self) -> int:
        return self._num_filters

    def build(self, input_shape: Tuple[int, int], num_classes: int) -> keras.models.Model:
        # Create the input layer
        input_layer = keras.layers.Input(input_shape)

        # Apply the convolution layers
        conv1 = keras.layers.Conv1D(filters=self.num_filters, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation('relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=2 * self.num_filters, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(filters=self.num_filters, kernel_size=3, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        # Pool the transformed result
        pooled = keras.layers.GlobalAveragePooling1D()(conv3)

        # Apply the output layer
        probs = keras.layers.Dense(num_classes, activation='softmax')(pooled)

        # Create and return the model
        model = keras.models.Model(inputs=input_layer, outputs=probs)

        return model
