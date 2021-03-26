from .base import BaseClassifier
from .mlp import MLP
from .resnet import ResNet
from .convnet import ConvNet


def make_model(name: str):
    if name == 'mlp':
        return MLP(batch_size=16,
                   learning_rate=1e-3,
                   train_frac=0.8,
                   hidden_units=512)
    elif name == 'conv':
        return ConvNet(batch_size=32,
                       learning_rate=1e-3,
                       train_frac=0.8,
                       num_filters=64)
    elif name == 'resnet':
        return ResNet(batch_size=64,
                      learning_rate=1e-3,
                      train_frac=0.8,
                      num_filters=64)
    else:
        raise ValueError('Unknown classifier: {0}'.format(name))


def restore_model(name: str, save_folder: str) -> BaseClassifier:
    model = make_model(name=name)
    model.restore(save_folder=save_folder)
    return model
