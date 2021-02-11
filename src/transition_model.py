import numpy as np
import h5py
from sklearn.linear_model import Ridge
from utils.file_utils import save_pickle_gz



def train(data_file: str, output_file: str):
    # Load the input data
    with h5py.File(data_file, 'r') as fin:
        dataset = fin['inputs'][:]  # [N, T, D]

    # Unpack the shape
    num_samples = dataset.shape[0]  # N
    seq_length = dataset.shape[1]  # T
    num_features = dataset.shape[2]  # D

    # Align samples for next-frame prediction
    input_list: List[np.ndarray] = []
    output_list: List[np.ndarray] = []

    for sample_idx in range(num_samples):
        seq_features = dataset[sample_idx]

        for seq_idx in range(seq_length - 1):
            input_list.append(np.expand_dims(seq_features[seq_idx], axis=0))
            output_list.append(np.expand_dims(seq_features[seq_idx + 1], axis=0))

    # Stack data into arrays
    inputs = np.vstack(input_list)  # [M, D]
    outputs = np.vstack(output_list)  # [M, D]

    # Fit the linear model
    data_mat = np.matmul(inputs.T, inputs)
    sol_mat = np.matmul(inputs.T, outputs)

    weights = np.linalg.solve(data_mat, sol_mat)

    preds = np.matmul(inputs, weights)  # [M, D]
    error = np.average(np.sum(np.square(preds - outputs), axis=-1))

    print('MSE: {0:.5f}'.format(error))

    save_pickle_gz(weights, output_file)


train(data_file='datasets/uci_har/validation/data.h5',
      output_file='transition_model.pkl.gz')
