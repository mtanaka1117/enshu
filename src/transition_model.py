import numpy as np
import h5py
import os.path
from argparse import ArgumentParser
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from utils.file_utils import save_pickle_gz, read_pickle_gz


WINDOW = 5

def train(data_file: str, scaler: StandardScaler, output_file: str):
    # Load the input data
    with h5py.File(data_file, 'r') as fin:
        dataset = fin['inputs'][:]  # [N, T, D]

    # Unpack the shape
    num_samples = dataset.shape[0]  # N
    seq_length = dataset.shape[1]  # T
    num_features = dataset.shape[2]  # D

    # Scale the data
    scaled_data = scaler.transform(dataset.reshape(num_samples * seq_length, num_features))
    dataset = scaled_data.reshape(num_samples, seq_length, num_features)

    # Align samples for next-frame prediction
    input_list: List[np.ndarray] = []
    output_list: List[np.ndarray] = []

    for sample_idx in range(num_samples):
        seq_features = dataset[sample_idx]

        for seq_idx in range(seq_length - 1):
            input_list.append(seq_features[seq_idx].reshape(1, -1))
            output_list.append(np.expand_dims(seq_features[seq_idx + 1], axis=0))

    # Stack data into arrays
    inputs = np.vstack(input_list)  # [M, D]
    outputs = np.vstack(output_list)  # [M, D]

    # Fit the linear model
    data_mat = np.matmul(inputs.T, inputs) + 0.01 * np.eye(inputs.shape[1])
    sol_mat = np.matmul(inputs.T, outputs)

    weights = np.linalg.solve(data_mat, sol_mat)

    preds = np.matmul(inputs, weights)  # [M, D]

    error = mean_squared_error(y_true=outputs, y_pred=preds)

    print('MSE: {0:.5f}'.format(error))
    print(weights)

    save_pickle_gz(weights, output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--inference-model', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()

    data_file = os.path.join(args.data_folder, 'validation', 'data.h5')

    inference_model = read_pickle_gz(args.inference_model)
    scaler = inference_model['scaler']

    train(data_file=data_file,
          scaler=scaler,
          output_file=args.output_file)
