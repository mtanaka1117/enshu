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
ALPHA = 0.1


def quantile_loss(preds: np.ndarray, outputs: np.ndarray, percentile: float):
    diff = outputs - preds
    return np.where(diff > 0, (1.0 - percentile) * diff, -1 * percentile * diff)


def quantile_loss_grad(preds: np.ndarray, outputs: np.ndarray, percentile: float):
    diff = outputs - preds
    return np.where(np.isclose(diff, 0), 0, np.where(diff > 0, percentile - 1.0, percentile))
    # return np.where(np.isclose(diff, 0), 0, np.where(diff > 0, percentile - 1.0, percentile))


def optimize(inputs: np.ndarray, outputs: np.ndarray, percentile: float, batch_size: int, max_iters: int):
    
    sample_idx = np.arange(inputs.shape[0])
    rand = np.random.RandomState(seed=92)

    weights = rand.uniform(low=-0.7, high=0.7, size=(inputs.shape[1], outputs.shape[1]))
    learning_rate = 0.1
    anneal_rate = 0.99

    for iter_idx in range(max_iters):
        batch_idx = rand.choice(sample_idx, size=batch_size, replace=False)
        batch_inputs = inputs[batch_idx, :]
        batch_outputs = outputs[batch_idx, :]

        preds = np.matmul(batch_inputs, weights)  # [B, D]

        sample_loss = np.sum(quantile_loss(preds, batch_outputs, percentile), axis=-1)  # [B]
        loss = np.average(sample_loss)
        
        loss_grad = quantile_loss_grad(preds, batch_outputs, percentile)  # [B, D]

        expanded_inputs = np.expand_dims(batch_inputs, axis=-1)  # [B, D, 1]
        expanded_grad = np.expand_dims(loss_grad, axis=1)  # [B, 1, D]

        weight_grad = np.matmul(expanded_inputs, expanded_grad)  # [B, D, D]
        weight_grad = np.average(weight_grad, axis=0)  # [D, D]

        weights = weights - learning_rate * weight_grad
        learning_rate *= anneal_rate

    preds = np.matmul(inputs, weights)
    loss = np.average(np.sum(quantile_loss(preds, outputs, percentile), axis=-1))

    return weights


class TransitionModel:

    def predict(self, x: np.ndarray) -> np.ndarray:

        if x.shape[0] == self._median.shape[0]:
            pred = np.matmul(x.reshape(1, -1), self._median).reshape(-1, 1)
            return pred

        return np.matmul(x, self._median)

    def confidence(self, x: np.ndarray) -> float:
        if x.shape[0] == self._lower.shape[0]:
            lower = np.matmul(x.reshape(1, -1), self._lower)
            upper = np.matmul(x.reshape(1, -1), self._upper)
        else:
            lower = np.matmul(x, self._lower)
            upper = np.matmul(x, self._upper) 

        return np.sum(np.abs(upper - lower))

    def train(self, data_file: str, scaler: StandardScaler, output_file: str):
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

        self._lower = optimize(inputs=inputs, outputs=outputs, batch_size=64, max_iters=200, percentile=0.1)
        self._median = optimize(inputs=inputs, outputs=outputs, batch_size=64, max_iters=200, percentile=0.5)
        self._upper = optimize(inputs=inputs, outputs=outputs, batch_size=64, max_iters=200, percentile=0.9)

        preds = self.predict(inputs)
        error = mean_squared_error(y_true=outputs, y_pred=preds)
        print('MSE: {0:.5f}'.format(error))

        models = {
            'lower': self._lower,
            'median': self._median,
            'upper': self._upper
        }

        save_pickle_gz(models, output_file)

    #    data_mat = np.matmul(inputs.T, inputs) + 0.01 * np.eye(inputs.shape[1])
    #    sol_mat = np.matmul(inputs.T, outputs)
#
#    weights = np.linalg.solve(data_mat, sol_mat)
#
#    preds = np.matmul(inputs, weights)  # [M, D]
#
#    error = mean_squared_error(y_true=outputs, y_pred=preds)
#
#    print('MSE: {0:.5f}'.format(error))
#    print(weights)
#

    @classmethod
    def restore(cls, path: str):
        serialized = read_pickle_gz(path)
        model = TransitionModel()

        model._lower = serialized['lower']
        model._median = serialized['median']
        model._upper = serialized['upper']

        return model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--inference-model', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()

    data_file = os.path.join(args.data_folder, 'validation', 'data.h5')

    inference_model = read_pickle_gz(args.inference_model)
    scaler = inference_model['scaler']

    transition_model = TransitionModel()

    transition_model.train(data_file=data_file,
                           scaler=scaler,
                           output_file=args.output_file)
