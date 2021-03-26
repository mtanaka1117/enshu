import numpy as np
import tensorflow.keras as keras
import os.path
from typing import Tuple

from .dnn import NeuralNetwork


def conv(inputs: keras.layers.Layer, num_filters: int, kernel_size: int, should_activate: bool) -> keras.layers.Layer:
    filtered = keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same')(inputs)
    norm = keras.layers.BatchNormalization()(filtered)
    return keras.layers.Activation('relu')(norm) if should_activate else norm


def residual_block(inputs: keras.layers.Layer, num_filters: int) -> keras.layers.Layer:
    # Apply the convolution transformations
    conv1 = conv(inputs=inputs, num_filters=num_filters, kernel_size=8, should_activate=True)
    conv2 = conv(inputs=conv1, num_filters=num_filters, kernel_size=5, should_activate=True)
    conv3 = conv(inputs=conv2, num_filters=num_filters, kernel_size=3, should_activate=False)

    # Compute the residual transformation
    residual = conv(inputs=inputs, num_filters=num_filters, kernel_size=1, should_activate=False)

    # Apply the residual
    output = keras.layers.add([residual, conv3])

    # Apply the non-linearity
    return keras.layers.Activation('relu')(output)


class ResNet(NeuralNetwork):

    def __init__(self, batch_size: int, train_frac: float, learning_rate: float, num_filters: int):
        super().__init__(batch_size=batch_size,
                         train_frac=train_frac,
                         learning_rate=learning_rate)
        self._num_filters = num_filters
 
    @property
    def name(self) -> str:
        return 'resnet'

    @property
    def num_filters(self) -> int:
        return self._num_filters

    def build(self, input_shape: Tuple[int, int], num_classes: int) -> keras.models.Model:
        # Create the input layer
        input_layer = keras.layers.Input(input_shape)

        # Apply the residual blocks
        block1 = residual_block(inputs=input_layer, num_filters=self.num_filters)
        block2 = residual_block(inputs=block1, num_filters=self.num_filters * 2)
        block3 = residual_block(inputs=block2, num_filters=self.num_filters * 2)

        # Pool the results
        pooled = keras.layers.GlobalAveragePooling1D()(block3)

        # Compute the probabilities
        probs = keras.layers.Dense(num_classes, activation='softmax')(pooled)

        # Compile the model
        model = keras.models.Model(inputs=input_layer, outputs=probs)

        return model
