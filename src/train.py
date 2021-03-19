import numpy as np
import h5py
from argparse import ArgumentParser
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

from policies import Policy, RandomPolicy, AdaptivePolicy
from utils.file_utils import save_pickle_gz


def train(train_file: str, val_file: str, output_file: str):
    # Read the input data
    with h5py.File(train_file, 'r') as fin:
        inputs = fin['inputs'][:]
        output = fin['output'][:]

    # Fit the data normalization object
    scaler = StandardScaler()
    model_inputs = scaler.fit_transform(inputs.reshape(-1, inputs.shape[-1]))  # [N * T, D]
    model_inputs = model_inputs.reshape(inputs.shape[0], -1)  # [N, T * D]
    model_output = output.reshape(-1)

    # Fit the inference model
    clf = MLPClassifier(hidden_layer_sizes=[64, 64], alpha=0.1, random_state=281, max_iter=500)
    clf.fit(model_inputs, model_output)

    # Evaluate the inference model
    print('Train Accuracy: {0:.5f}'.format(clf.score(model_inputs, model_output)))

    with h5py.File(val_file, 'r') as fval:
        val_inputs = fval['inputs'][:]
        val_output = fval['output'][:]

    val_input_shape = val_inputs.shape
    val_inputs = scaler.transform(val_inputs.reshape(-1, val_input_shape[-1]))
    val_inputs = val_inputs.reshape(val_input_shape[0], -1)
    val_output = val_output.reshape(-1)

    print('Validation Accuracy: {0:.5f}'.format(clf.score(val_inputs, val_output)))

    # Save the results
    result_dict = {
        'model': clf,
        'scaler': scaler
    }
    
    save_pickle_gz(result_dict, output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    args = parser.parse_args()

    train(train_file='datasets/{0}/train/data.h5'.format(args.dataset_name),
          val_file='datasets/{0}/validation/data.h5'.format(args.dataset_name),
          output_file='saved_models/{0}/model.pkl.gz'.format(args.dataset_name))
