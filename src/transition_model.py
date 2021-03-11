import numpy as np
import h5py
import os.path
import tensorflow as tf
from argparse import ArgumentParser
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from utils.data_utils import apply_dropout
from utils.file_utils import save_pickle_gz, read_pickle_gz
from utils.constants import BIG_NUMBER


class TransitionModel:

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

        self._fit(inputs=inputs, outputs=outputs, output_file=output_file)

    @classmethod
    def restore(cls, path: str):
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    def confidence(self, x: np.ndarray) -> float:
        pass

    def _fit(self, inputs: np.ndarray, outputs: np.ndarray, output_file: str):
        """
        Fits the dropout model to the given dataset.

        Args:
            inputs: A [N, D] array of input features
            outputs: A [N, D] array of output features
        """
        pass


class LinearModel(TransitionModel):

    def _fit(self, inputs: np.ndarray, outputs: np.ndarray, output_file: str):
        # Fit the linear model
        data_mat = np.matmul(inputs.T, inputs) + 0.01 * np.eye(inputs.shape[1])
        sol_mat = np.matmul(inputs.T, outputs)

        self._weights = np.linalg.solve(data_mat, sol_mat)  # [D, D]

        preds = np.matmul(inputs, self._weights)  # [M, D]
        error = mean_squared_error(y_true=outputs, y_pred=preds)

        save_pickle_gz(self._weights, output_file)

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 2, 'Must pass in a 2d array'
        return np.matmul(x, self._weights)  # [K, D]

    def confidence(self, x: np.ndarray) -> float:
        return 0.0  # TODO: Replace with a prediction interval

    @classmethod
    def restore(cls, path: str):
        weights = read_pickle_gz(path)

        model = cls()
        model._weights = weights

        return model


class DropoutModel(TransitionModel):

    def __init__(self):
        self._sess = tf.compat.v1.Session(graph=tf.Graph())
        self._dropout_rate = 0.5
        self._lr = 0.001
        self._batch_size = 8
        self._num_epochs = 10
        self._patience = 2
        self._confidence_iters = 6
        
        self._rand = np.random.RandomState(seed=395)

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 2, 'Must have a 2d input'
        weights = apply_dropout(self._weight_mat, drop_rate=self._dropout_rate, rand=self._rand)

        return np.matmul(x, weights) + self._bias_vec

    def confidence(self, x: np.ndarray) -> float:
        preds = np.concatenate([self.predict(x=x) for _ in range(self._confidence_iters)], axis=0)  # [K, D]

        mean_pred = np.average(preds, axis=0)
        var_pred = np.average(np.sum(np.square(preds - np.expand_dims(mean_pred, axis=0)), axis=-1))

        return var_pred

    def _build(self, num_features: int):
        with self._sess.graph.as_default():

            tf.random.set_seed(seed=196)

            # Make the placeholders
            self._inputs_ph = tf.compat.v1.placeholder(shape=[None, num_features],
                                                       dtype=tf.float32,
                                                       name='inputs-ph')
            self._outputs_ph = tf.compat.v1.placeholder(shape=[None, num_features],
                                                        dtype=tf.float32,
                                                        name='outputs-ph')

            # Make the trainable variables
            self._weights = tf.compat.v1.get_variable(name='weights',
                                                      shape=[num_features, num_features],
                                                      dtype=tf.float32,
                                                      initializer=tf.compat.v1.glorot_uniform_initializer())
            self._bias = tf.compat.v1.get_variable(name='bias',
                                                   shape=[1, num_features],
                                                   dtype=tf.float32,
                                                   initializer=tf.compat.v1.glorot_uniform_initializer())

            # Compute the prediction
            weights = tf.nn.dropout(self._weights, rate=self._dropout_rate)  # [D, D]
            self._pred = tf.matmul(self._inputs_ph, weights) + self._bias  # [N, D]

            # Make the loss
            self._loss = tf.reduce_mean(tf.square(self._pred - self._outputs_ph))  # Scalar

            # Make the optimization step
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self._lr)

            variables = [self._weights, self._bias]
            gradients = tf.gradients(self._loss, variables)
            pruned_gradients = [(grad, var) for grad, var in zip(gradients, variables) if grad is not None]
            self._train_step = optimizer.apply_gradients(pruned_gradients)
    
            # Initialize the variables
            self._sess.run(tf.compat.v1.global_variables_initializer())

    def _fit(self, inputs: np.ndarray, outputs: np.ndarray, output_file: str):
        # Build the model
        self._build(num_features=inputs.shape[-1])

        sample_idx = np.arange(inputs.shape[0])
        num_samples = len(sample_idx)

        best_loss = BIG_NUMBER
        early_stop_counter = 0

        with self._sess.graph.as_default():
            for epoch in range(self._num_epochs):

                print('==========')
                print('Epoch {0}'.format(epoch))
                print('==========')

                # Shuffle the sample indices
                self._rand.shuffle(sample_idx)

                losses: List[float] = []

                for batch_idx in range(0, num_samples, self._batch_size):

                    # Make the batch
                    start, end = batch_idx, batch_idx + self._batch_size
                    batch_inputs = inputs[start:end]
                    batch_outputs = outputs[start:end]

                    # Apply the optimization step
                    feed_dict = {
                        self._inputs_ph: batch_inputs,
                        self._outputs_ph: batch_outputs
                    }

                    results = self._sess.run([self._loss, self._train_step], feed_dict=feed_dict)

                    losses.append(results[0])
                    
                    print('Loss So Far: {0:.4f}'.format(np.average(losses)), end='\r')

                print()

                # Record aggregate results
                epoch_loss = np.average(losses)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    early_stop_counter = 0
                    self._save(output_file)
                else:
                    early_stop_counter += 1

                if early_stop_counter > self._patience:
                    print('Early Stopping.')
                    break

    def _save(self, output_file: str):
        with self._sess.graph.as_default():

            # Extract the trainable variables
            trainable_vars = list(self._sess.graph.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES))
            var_dict = {var.name: var for var in trainable_vars}
            var_result = self._sess.run(var_dict)

            save_pickle_gz(var_result, output_file)

    @classmethod
    def restore(cls, output_file: str):
        serialized = read_pickle_gz(output_file)

        model = cls()
        model._weight_mat = serialized['weights:0']
        model._bias_vec = serialized['bias:0']

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

    model = DropoutModel.restore('dropout_model.pkl.gz')

    x = np.array([-1.0, 1.0]).reshape(1, -1)
    print(model.confidence(x=x))

    #transition_model = DropoutModel()

    #transition_model.train(data_file=data_file,
    #                       scaler=scaler,
    #                       output_file=args.output_file)
