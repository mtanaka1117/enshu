import os.path
import h5py
import numpy as np
from argparse import ArgumentParser

from classifiers import restore_model
from utils.file_utils import read_pickle_gz


def test(dataset_path: str, model_folder: str, model_name: str):
    # Load the data
    with h5py.File(dataset_path, 'r') as fin:
        inputs = fin['inputs'][:]
        output = fin['output'][:]

    if len(inputs.shape) == 2:
        inputs = np.expand_dims(inputs, axis=-1)

    # Make the model
    model = restore_model(name=model_name, save_folder=model_folder)

    # Normalize the data
    scaler_path = os.path.join(model_folder, '{0}_scaler.pkl.gz'.format(model.name))
    scaler = read_pickle_gz(scaler_path)

    input_shape = inputs.shape
    inputs = scaler.transform(inputs.reshape(-1, input_shape[-1]))
    inputs = inputs.reshape(input_shape)
    output = output.reshape(-1)

    # Test the model
    accuracy = model.accuracy(inputs=inputs, labels=output)
    print('Test Accuracy: {0:.5f}'.format(accuracy))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-type', type=str, required=True, choices=['mlp', 'resnet', 'convnet', 'temporal'])
    parser.add_argument('--dataset-name', type=str, required=True)
    args = parser.parse_args()

    dataset_path = os.path.join('datasets', args.dataset_name, 'test', 'data.h5')
    model_folder = os.path.join('saved_models', args.dataset_name)

    test(dataset_path=dataset_path,
         model_folder=model_folder,
         model_name=args.model_type)
