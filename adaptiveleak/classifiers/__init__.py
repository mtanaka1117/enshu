from .base import BaseClassifier
from .mlp import MLP
from .resnet import ResNet
from .convnet import ConvNet


def make_model(name: str):
    if name == 'mlp':
        return MLP(batch_size=16,
                   hidden_units=512)
    elif name == 'conv':
        return ConvNet(batch_size=16,
                       num_filters=128)
    elif name == 'resnet':
        return ResNet(batch_size=64,
                      num_filters=128)
    else:
        raise ValueError('Unknown classifier: {0}'.format(name))


def restore_model(name: str, save_folder: str) -> BaseClassifier:
    model = make_model(name=name)
    model.restore(save_folder=save_folder)
    return model
