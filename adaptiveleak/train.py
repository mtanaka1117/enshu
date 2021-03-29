import numpy as np
import h5py
import os.path
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler

from classifiers import make_model
from utils.file_utils import save_pickle_gz
from test import test


def train(model_name: str, train_file: str, val_file: str, save_folder: str, num_epochs: int):
    # Read the input data
    with h5py.File(train_file, 'r') as fin:
        train_inputs = fin['inputs'][:]
        train_output = fin['output'][:]

    # Ensure all inputs are 3d
    if len(train_inputs.shape) == 2:
        train_inputs = np.expand_dims(train_inputs, axis=-1)

    # Fit the data normalization object
    scaler = StandardScaler()

    train_shape = train_inputs.shape
    train_inputs = scaler.fit_transform(train_inputs.reshape(-1, train_shape[-1]))  # [N * T, D]
    train_inputs = train_inputs.reshape(train_shape)  # [N, T, D]

    train_output = train_output.reshape(-1)

    # Read and scale the validation set
    with h5py.File(val_file, 'r') as fval:
        val_inputs = fval['inputs'][:]
        val_output = fval['output'][:]

    # Ensure all inputs are 3d
    if len(val_inputs.shape) == 2:
        val_inputs = np.expand_dims(val_inputs, axis=-1)

    val_shape = val_inputs.shape
    val_inputs = scaler.transform(val_inputs.reshape(-1, val_shape[-1]))
    val_inputs = val_inputs.reshape(val_shape)
    val_output = val_output.reshape(-1)

    # Fit the inference model
    clf = make_model(name=model_name)

    print('Fitting model...')
    clf.fit(train_inputs=train_inputs,
            train_labels=train_output,
            val_inputs=val_inputs,
            val_labels=val_output,
            num_epochs=num_epochs,
            save_folder=save_folder)

    # Save the results
    scaler_file = os.path.join(save_folder, '{0}_scaler.pkl.gz'.format(clf.name))
    save_pickle_gz(scaler, scaler_file)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-type', type=str, required=True, choices=['mlp', 'resnet', 'conv', 'temporal'])
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--num-epochs', type=int, required=True)
    args = parser.parse_args()

    train_path = os.path.join('datasets', args.dataset_name, 'train', 'data.h5')
    val_path = os.path.join('datasets', args.dataset_name, 'validation', 'data.h5')
    save_folder = os.path.join('saved_models', args.dataset_name)

    train(train_file=train_path,
          val_file=val_path,
          save_folder=save_folder,
          model_name=args.model_type,
          num_epochs=args.num_epochs)

    test_path = os.path.join('datasets', args.dataset_name, 'test', 'data.h5')
    test(dataset_path=test_path, model_folder=save_folder, model_name=args.model_type)
