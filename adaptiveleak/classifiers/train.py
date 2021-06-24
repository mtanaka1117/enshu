import h5py
import numpy as np
import os.path

from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler

from resnet import ResNet
from adaptiveleak.utils.file_utils import save_pickle_gz


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    # Load the training data
    train_path = os.path.join('..', 'datasets', args.dataset, 'train', 'data.h5')
    with h5py.File(train_path, 'r') as fin:
        train_inputs = fin['inputs'][:]  # [N, T, D]
        train_labels = fin['output'][:]  # [N]

    if len(train_inputs.shape) == 2:
        train_inputs = np.expand_dims(train_inputs, axis=-1)

    # Load the validation data
    val_path = os.path.join('..', 'datasets', args.dataset, 'validation', 'data.h5')
    with h5py.File(val_path, 'r') as fin:
        val_inputs = fin['inputs'][:]  # [M, T, D]
        val_labels = fin['output'][:]  # [M]

    if len(val_inputs.shape) == 2:
        val_inputs = np.expand_dims(val_inputs, axis=-1)

    # Make the model
    model = ResNet(batch_size=16, num_filters=2)

    # Scale the inputs
    scaler = StandardScaler()

    train_shape = train_inputs.shape
    train_inputs = scaler.fit_transform(train_inputs.reshape(-1, train_shape[-1]))
    train_inputs = train_inputs.reshape(train_shape)

    val_shape = val_inputs.shape
    val_inputs = scaler.transform(val_inputs.reshape(-1, val_shape[-1]))
    val_inputs = val_inputs.reshape(val_shape)

    # Train the model
    save_folder = os.path.join('..', 'saved_models', args.dataset)

    model.fit(train_inputs=train_inputs,
              train_labels=train_labels,
              val_inputs=val_inputs,
              val_labels=val_labels,
              num_epochs=1,
              save_folder=save_folder)

    # Save the data scaling object
    scaling_file = os.path.join(save_folder, 'classifier_scaler.pkl.gz')
    save_pickle_gz(scaler, scaling_file)
