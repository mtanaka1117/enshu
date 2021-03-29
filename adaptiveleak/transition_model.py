import numpy as np
import h5py
import os.path
import tensorflow as tf
import scipy.optimize as opt
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from argparse import ArgumentParser
from functools import partial
from collections import namedtuple
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from utils.data_utils import apply_dropout, leaky_relu, softmax
from utils.file_utils import save_pickle_gz, read_pickle_gz
from utils.constants import BIG_NUMBER, LINEAR_TRANSITION, DROPOUT_TRANSITION, INTERVAL_TRANSITION, BOOTSTRAP_TRANSITION, QUANTILE_TRANSITION


Placeholders = namedtuple('Placeholders', ['inputs', 'outputs', 'dropout'])
Operations = namedtuple('Operations', ['pred', 'loss', 'train_step'])


class TransitionModel:

    def train(self, data_file: str, scaler: StandardScaler, output_folder: str):
        # Load the input data
        with h5py.File(data_file, 'r') as fin:
            dataset = fin['inputs'][:]  # [N, T, D]

        if len(dataset.shape) == 2:
            dataset = np.expand_dims(dataset, axis=-1)

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

        output_file = os.path.join(output_folder, '{0}.pkl.gz'.format(self.name))
        self._fit(inputs=inputs, outputs=outputs, output_file=output_file)

    @property
    def name(self) -> str:
        raise NotImplementedError()

    @classmethod
    def restore(cls, path: str):
        raise NotImplementedError()

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def confidence(self, x: np.ndarray) -> float:
        raise NotImplementedError()

    def _fit(self, inputs: np.ndarray, outputs: np.ndarray, output_file: str):
        """
        Fits the dropout model to the given dataset.

        Args:
            inputs: A [N, D] array of input features
            outputs: A [N, D] array of output features
        """
        raise NotImplementedError()


class LinearModel(TransitionModel):

    @property
    def name(self) -> str:
        return LINEAR_TRANSITION

    def _fit(self, inputs: np.ndarray, outputs: np.ndarray, output_file: str):
        # Fit the linear model
        data_mat = np.matmul(inputs.T, inputs) + 0.01 * np.eye(inputs.shape[1])
        sol_mat = np.matmul(inputs.T, outputs)

        self._weights = np.linalg.solve(data_mat, sol_mat)  # [D, D]

        preds = np.matmul(inputs, self._weights)  # [M, D]
        error = mean_squared_error(y_true=outputs, y_pred=preds)

        print('MSE: {0:.4f}'.format(error))

        save_pickle_gz(self._weights, output_file)

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 2, 'Must pass in a 2d array'

        if x.shape[0] == self._weights.shape[0]:
            return np.matmul(self._weights.T, x)  # [D, K]

        return np.matmul(x, self._weights)  # [K, D]

    def confidence(self, x: np.ndarray) -> float:
        return 0.0  # TODO: Replace with a prediction interval

    @classmethod
    def restore(cls, path: str):
        weights = read_pickle_gz(path)

        model = cls()
        model._weights = weights

        return model


class QuantileModel(TransitionModel):

    def __init__(self):
        self._rand = np.random.RandomState(seed=548)

    @property
    def name(self) -> str:
        return QUANTILE_TRANSITION

    def _fit(self, inputs: np.ndarray, outputs: np.ndarray, output_file: str):
        # Fit the linear model
        data_mat = np.matmul(inputs.T, inputs) + 0.01 * np.eye(inputs.shape[1])
        sol_mat = np.matmul(inputs.T, outputs)

        self._weights = np.linalg.solve(data_mat, sol_mat)  # [D, D]

        # Create the loss function
        def loss_fn(w: np.ndarray, X: np.ndarray, y: np.ndarray, quantile: float):
            pred = np.matmul(X, w)  # [N]
            diff = y - pred  # [N]

            sample_loss = np.where(diff > 0, quantile * diff, (quantile - 1) * diff)  # [N]
            return np.average(sample_loss)  # Scalar

        # Optimize the upper and lower models
        upper_list: List[np.ndarray] = []
        lower_list: List[np.ndarray] = []

        num_features = inputs.shape[-1]
        w0 = self._rand.normal(loc=0.0, scale=1.0, size=(num_features, ))

        for feature_idx in range(num_features):
            loss_upper = partial(loss_fn, X=inputs, y=outputs[:, feature_idx], quantile=0.95)
            loss_lower = partial(loss_fn, X=inputs, y=outputs[:, feature_idx], quantile=0.05)

            res_upper = opt.minimize(method='L-BFGS-B',
                                     fun=loss_upper,
                                     x0=w0)

            res_lower = opt.minimize(method='L-BFGS-B',
                                     fun=loss_lower,
                                     x0=w0)

            upper_list.append(np.expand_dims(res_upper.x, axis=0))
            lower_list.append(np.expand_dims(res_lower.x, axis=0))

        # Stack the upper and lower weights
        self._upper = np.vstack(upper_list)
        self._lower = np.vstack(lower_list)

        preds = np.matmul(inputs, self._weights)  # [M, D]
        error = mean_squared_error(y_true=outputs, y_pred=preds)  # Scalar

        print('MSE: {0:.4f}'.format(error))

        #abs_error = np.sum(np.abs(preds - outputs), axis=-1)  # [M]

        #higher_pred = np.matmul(inputs, self._upper)  # [M, D]
        #lower_pred = np.matmul(inputs, self._lower)  # [M, D]

        #intervals = np.sum(np.abs(higher_pred - lower_pred), axis=-1)

        #plt.scatter(intervals, abs_error)
        #plt.show()

        params = {
            'weights': self._weights,
            'lower': self._lower,
            'upper': self._upper
        }
        save_pickle_gz(params, output_file)

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 2, 'Must pass in a 2d array'

        if x.shape[0] == self._weights.shape[0]:
            return np.matmul(self._weights.T, x)  # [D, K]

        return np.matmul(x, self._weights)  # [K, D]

    def confidence(self, x: np.ndarray) -> float:
        assert len(x.shape) == 2 and (x.shape[1] == 1), 'Expected a [D, 1] array'

        x = x.T
        lower_pred = np.matmul(x, self._lower)  # [1, D]
        upper_pred = np.matmul(x, self._upper)  # [1, D]

        abs_diff = np.abs(upper_pred - lower_pred)
        return np.sum(abs_diff)

    @classmethod
    def restore(cls, path: str):
        serialized = read_pickle_gz(path)

        model = cls()
        model._weights = serialized['weights']
        model._lower = serialized['lower']
        model._upper = serialized['upper']

        return model


class BootstrapModel(TransitionModel):

    def __init__(self):
        self._n_estimators = 4
        self._rand = np.random.RandomState(seed=192)

    @property
    def name(self) -> str:
        return BOOTSTRAP_TRANSITION

    def _fit(self, inputs: np.ndarray, outputs: np.ndarray, output_file: str):
        num_samples = inputs.shape[0]
        sample_idx = np.arange(num_samples)  # [N]

        weights: List[np.ndarray] = []

        # Fit each model
        for model_idx in range(self._n_estimators):
            bootstrap_idx = self._rand.choice(sample_idx, size=num_samples, replace=True)
            model_inputs = inputs[bootstrap_idx]
            model_outputs = outputs[bootstrap_idx]

            data_mat = np.matmul(model_inputs.T, model_inputs) + 0.01 * np.eye(model_inputs.shape[1])
            sol_mat = np.matmul(model_inputs.T, model_outputs)

            model_weights = np.linalg.solve(data_mat, sol_mat)  # [D, D]
        
            weights.append(model_weights)

        # Serialize the results
        save_pickle_gz(weights, output_file) 

    def _predict_multiple(self, x: np.ndarray) -> List[np.ndarray]:
        assert len(x.shape) == 2, 'Must pass in a 2d array'

        pred_list: List[np.ndarray] = []

        for weights in self._weights:
            if x.shape[0] == weights.shape[0]:
                pred = np.matmul(weights.T, x)  # [D, N]
            else:
                pred = np.matmul(x, weights)  # [N, D]

            pred_list.append(pred.reshape(x.shape))

        return pred_list

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred_list = self._predict_multiple(x=x)
        preds = np.vstack([np.expand_dims(a, axis=0) for a in pred_list])
        return np.average(preds, axis=0).reshape(x.shape)  # [N, D]
        
    def confidence(self, x: np.ndarray) -> float:
        assert x.shape[1] == 1, 'Must pass a [D, 1] array'

        # Compute the prediction for each model
        pred_list = self._predict_multiple(x=x)  # List of L [D, 1] arrays
        preds = np.vstack([np.expand_dims(a, axis=0) for a in pred_list])  # [L, D, 1]
        
        # Get the mean prediction
        mean = np.expand_dims(np.average(preds, axis=0), axis=0)  # [1, D, 1]

        square_diff = np.sum(np.squeeze(np.square(preds - mean), axis=-1), axis=-1)  # [L]
        return np.average(square_diff) * 100

    @classmethod
    def restore(cls, path: str):
        serialized = read_pickle_gz(path)

        model = cls()
        model._weights = serialized

        return model


class NeuralNetworkModel(TransitionModel):

    def __init__(self, hidden_units: int):
        self._sess = tf.compat.v1.Session(graph=tf.Graph())
        self._dropout_rate = 0.3
        self._lr = 0.001
        self._batch_size = 64
        self._num_epochs = 10
        self._patience = 2
        self._hidden_units = hidden_units
        self._train_frac = 0.8

        self._rand = np.random.RandomState(seed=395)
        
        self._phs = Placeholders(inputs=None, outputs=None, dropout=None)
        self._ops = Operations(pred=None, loss=None, train_step=None)
        self._metadata: Dict[str, Any] = dict()

    def _make_graph(self, inputs: tf.compat.v1.placeholder, num_features: int) -> tf.Tensor:
        raise NotImplementedError()

    def _preprocess(self, inputs: np.ndarray):
        pass

    def _make_placeholders(self, num_features: int) -> Placeholders:
        inputs_ph = tf.compat.v1.placeholder(shape=[None, num_features],
                                             dtype=tf.float32,
                                             name='inputs-ph')
        outputs_ph = tf.compat.v1.placeholder(shape=[None, num_features],
                                              dtype=tf.float32,
                                              name='outputs-ph')
        dropout_ph = tf.compat.v1.placeholder(shape=(),
                                              dtype=tf.float32,
                                              name='dropout-ph')
        return Placeholders(inputs=inputs_ph, outputs=outputs_ph, dropout=dropout_ph)

    def _make_loss(self, pred: tf.Tensor, expected: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.square(pred - expected))  # Scalar

    def _make_train_step(self, loss: tf.Tensor) -> tf.Tensor:
        # Make the optimization step
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self._lr)

        # Fetch all the trainable variables
        variables = list(self._sess.graph.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES))

        # Compute the gradients with respect to the loss
        gradients = tf.gradients(loss, variables)
        pruned_gradients = [(grad, var) for grad, var in zip(gradients, variables) if grad is not None]

        # Return the update step
        return optimizer.apply_gradients(pruned_gradients)
    
    def _make(self, num_features: int):
        with self._sess.graph.as_default():
            # Set seed for reproducible results
            tf.random.set_seed(seed=196)

            # Make the model placeholders
            self._phs = self._make_placeholders(num_features=num_features)

            # Make the computation graph
            pred = self._make_graph(inputs=self._phs.inputs,
                                    dropout=self._phs.dropout,
                                    num_features=num_features)

            # Make the loss function
            loss = self._make_loss(pred=pred, expected=self._phs.outputs)

            # Make the training step
            train_step = self._make_train_step(loss=loss)

            # Collect the operations
            self._ops = Operations(pred=pred, loss=loss, train_step=train_step)            

            # Initialize the variables
            self._sess.run(tf.compat.v1.global_variables_initializer())

    def _fit(self, inputs: np.ndarray, outputs: np.ndarray, output_file: str):
        # Perform any data pre-processing
        self._preprocess(inputs=inputs)

        # Build the model
        self._make(num_features=inputs.shape[-1])

        sample_idx = np.arange(inputs.shape[0])
        num_samples = len(sample_idx)

        # Split into train and validation sets
        split_idx = int(self._train_frac * num_samples)
        train_idx, val_idx = sample_idx[:split_idx], sample_idx[split_idx:]

        best_loss = BIG_NUMBER
        early_stop_counter = 0

        with self._sess.graph.as_default():
            for epoch in range(self._num_epochs):

                print('==========')
                print('Epoch {0}'.format(epoch))
                print('==========')

                # Shuffle the sample indices
                self._rand.shuffle(train_idx)
                self._rand.shuffle(val_idx)

                train_losses: List[float] = []
                val_losses: List[float] = []

                for idx, batch_start in enumerate(range(0, len(train_idx), self._batch_size)):

                    # Make the batch
                    start, end = batch_start, batch_start + self._batch_size
                    batch_idx = train_idx[start:end]

                    batch_inputs = inputs[batch_idx]
                    batch_outputs = outputs[batch_idx]

                    # Apply the optimization step
                    feed_dict = {
                        self._phs.inputs: batch_inputs,
                        self._phs.outputs: batch_outputs,
                        self._phs.dropout: self._dropout_rate
                    }

                    results = self._sess.run([self._ops.loss, self._ops.train_step], feed_dict=feed_dict)

                    train_losses.append(results[0])
                   
                    if (idx % 100) == 0:
                        print('Train Batch: {0}, Loss So Far: {1:.4f}'.format(idx, np.average(train_losses)), end='\r')

                print()

                for idx, batch_start in enumerate(range(0, len(val_idx), self._batch_size)):

                    # Make the batch
                    start, end = batch_start, batch_start + self._batch_size
                    batch_idx = val_idx[start:end]

                    batch_inputs = inputs[batch_idx]
                    batch_outputs = outputs[batch_idx]

                    # Apply the optimization step
                    feed_dict = {
                        self._phs.inputs: batch_inputs,
                        self._phs.outputs: batch_outputs,
                        self._phs.dropout: 0.0
                    }

                    results = self._sess.run(self._ops.loss, feed_dict=feed_dict)

                    val_losses.append(results)
                   
                    if (idx % 100) == 0:
                        print('Val Batch: {0}, Loss So Far: {1:.4f}'.format(idx, np.average(val_losses)), end='\r')

                print()

                # Record aggregate results
                epoch_loss = np.average(val_losses)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    early_stop_counter = 0
                    
                    print('Saving...')
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

            # Add any additional meta-data
            var_result.update(**self._metadata)

            # Save results into a pickle file
            save_pickle_gz(var_result, output_file)


class DropoutModel(NeuralNetworkModel):

    def __init__(self, hidden_units: int):
        super().__init__(hidden_units=hidden_units)
        self._confidence_iters = 4
        self._dropout_rate = 0.1

    @property
    def name(self) -> str:
        return DROPOUT_TRANSITION

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 2, 'Must have a 2d input'

        input_shape = x.shape

        if input_shape[0] == self._W1.shape[0]:
            x = np.transpose(x)

        hidden = leaky_relu(np.matmul(x, self._W1) + self._b1, alpha=0.25)

        pred = np.matmul(hidden, self._W2) + self._b2
        return pred.reshape(input_shape)

    def confidence(self, x: np.ndarray) -> float:
        assert len(x.shape) == 2 and x.shape[1] == 1, 'Must provide a (D, 1) input.'
        
        x = np.transpose(x)  # [1, D]
        hidden = leaky_relu(np.matmul(x, self._W1) + self._b1, alpha=0.25)

        pred_list: List[np.ndarray] = []

        for _ in range(self._confidence_iters):
            hidden_dropout = apply_dropout(hidden, drop_rate=self._dropout_rate, rand=self._rand)
            pred = np.matmul(hidden_dropout, self._W2) + self._b2
            pred_list.append(pred)

        preds = np.vstack(pred_list)  # [K, D]

        mean_pred = np.expand_dims(np.average(preds, axis=0), axis=0)
        var_pred = np.average(np.sum(np.square(preds - mean_pred), axis=-1))

        return var_pred

    def _make_graph(self, inputs: tf.Tensor, dropout: tf.Tensor, num_features: int):
        # Make the trainable variables
        W1 = tf.compat.v1.get_variable(name='W1',
                                       shape=[num_features, self._hidden_units],
                                       dtype=tf.float32,
                                       initializer=tf.compat.v1.glorot_uniform_initializer())
        b1 = tf.compat.v1.get_variable(name='b1',
                                       shape=[1, self._hidden_units],
                                       dtype=tf.float32,
                                       initializer=tf.compat.v1.glorot_uniform_initializer())

        W2 = tf.compat.v1.get_variable(name='W2',
                                       shape=[self._hidden_units, num_features],
                                       dtype=tf.float32,
                                       initializer=tf.compat.v1.glorot_uniform_initializer())
        b2 = tf.compat.v1.get_variable(name='b2',
                                       shape=[1, num_features],
                                       dtype=tf.float32,
                                       initializer=tf.compat.v1.glorot_uniform_initializer())

        # Apply the neural network
        hidden = tf.nn.leaky_relu(tf.matmul(inputs, W1) + b1, alpha=0.25)  # [N, K]
        hidden_dropout = tf.nn.dropout(hidden, rate=dropout)  # [N, K]

        pred = tf.matmul(hidden_dropout, W2) + b2  # [N, D]
        return pred

    @classmethod
    def restore(cls, output_file: str):
        serialized = read_pickle_gz(output_file)

        hidden_units = serialized['W1:0'].shape[-1]

        model = cls(hidden_units=hidden_units)
        model._W1 = serialized['W1:0']
        model._b1 = serialized['b1:0']
        model._W2 = serialized['W2:0']
        model._b2 = serialized['b2:0']

        return model


class IntervalModel(NeuralNetworkModel):

    def __init__(self, hidden_units: int, num_bins: int):
        super().__init__(hidden_units=hidden_units)
        self._num_bins = num_bins
        self._alpha = 0.3

    @property
    def name(self) -> str:
        return INTERVAL_TRANSITION

    def predict(self, x: np.ndarray) -> np.ndarray:
        input_shape = x.shape
        pred = self._predict_probs(x=x)[0]
        return pred.reshape(input_shape)

    def _predict_probs(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 2, 'Must have a 2d input'

        # Reshape if features on the first axis
        if x.shape[-1] == 1:
            x = np.transpose(x)
        
        num_samples = x.shape[0]

        hidden = leaky_relu(np.matmul(x, self._W1) + self._b1, alpha=0.25)  # [N, K]
        pred_logits = np.matmul(hidden, self._W2) + self._b2  # [N, D * L]
        pred_logits = pred_logits.reshape(num_samples, -1, self._num_bins)  # [N, D, L]

        pred_probs = softmax(pred_logits, axis=-1)

        bins = np.expand_dims(self._bins, axis=0)  # [1, D, L]
        pred = np.sum(pred_probs * bins, axis=-1)  # [N, D]

        return pred, pred_probs

    def confidence(self, x: np.ndarray) -> float:
        # Get the prediction, [N, D] and [N, D, L] arrays
        pred, probs = self._predict_probs(x=x)

        # Get the bin index for each prediction
        upper_bins = self._bin_endpoints[:, 1:]  # [D, L]
        lower_bins = self._bin_endpoints[:, :-1]  # [D, L]

        intervals: List[float] = []

        for sample_idx in range(pred.shape[0]):
            for feature_idx in range(pred.shape[1]):
                sample_pred = pred[sample_idx, feature_idx]  # Scalar
                sample_probs = probs[sample_idx, feature_idx]  # [L]

                feature_upper_bins = upper_bins[feature_idx]
                feature_lower_bins = lower_bins[feature_idx]

                # Get the bin index
                bin_idx = 0
                while bin_idx < self._num_bins and (sample_pred < feature_lower_bins[bin_idx] or sample_pred > feature_upper_bins[bin_idx]):
                    bin_idx += 1

                current_prob = sample_probs[bin_idx]
                offset = 1

                while current_prob < self._alpha:
                    current_prob += sample_probs[bin_idx - offset] if bin_idx >= offset else 0
                    current_prob += sample_probs[bin_idx + offset] if bin_idx + offset < self._num_bins else 0

                    offset += 1

                upper_limit = min(self._num_bins - 1, bin_idx + offset)
                lower_limit = max(0, bin_idx - offset)

                intervals.append(upper_limit - lower_limit)

        return np.average(intervals)

    def _preprocess(self, inputs: np.ndarray):
        # Get evenly divided percentiles
        percentiles = np.linspace(start=0.0, stop=1.0, num=self._num_bins + 1) * 100.0

        bin_endpoints = np.percentile(inputs, q=percentiles, axis=0)  # [L + 1, D]
        bin_endpoints = np.transpose(bin_endpoints)  # [D, L + 1]

        min_endpoints = bin_endpoints[:, :-1]  # [D, L]
        max_endpoints = bin_endpoints[:, 1:]  # [D, L]

        bins = (max_endpoints + min_endpoints) / 2  # [D, L]

        self._bin_endpoints = bin_endpoints
        self._bins = bins

        self._metadata['bins'] = self._bins
        self._metadata['bin_endpoints'] = self._bin_endpoints

    def _make_graph(self, inputs: tf.Tensor, num_features: int):
        # Make the trainable variables
        W1 = tf.compat.v1.get_variable(name='W1',
                                       shape=[num_features, self._hidden_units],
                                       dtype=tf.float32,
                                       initializer=tf.compat.v1.glorot_uniform_initializer())
        b1 = tf.compat.v1.get_variable(name='b1',
                                       shape=[1, self._hidden_units],
                                       dtype=tf.float32,
                                       initializer=tf.compat.v1.glorot_uniform_initializer())

        W2 = tf.compat.v1.get_variable(name='W2',
                                       shape=[self._hidden_units, num_features * self._num_bins],
                                       dtype=tf.float32,
                                       initializer=tf.compat.v1.glorot_uniform_initializer())
        b2 = tf.compat.v1.get_variable(name='b2',
                                       shape=[1, num_features * self._num_bins],
                                       dtype=tf.float32,
                                       initializer=tf.compat.v1.glorot_uniform_initializer())

        # Apply the neural network
        hidden = tf.nn.leaky_relu(tf.matmul(inputs, W1) + b1, alpha=0.25)  # [N, K]

        pred_logits = tf.matmul(hidden, W2) + b2  # [N, D * L]
        pred_logits = tf.reshape(pred_logits, (-1, num_features, self._num_bins))  # [N, D, L]

        pred_probs = tf.nn.softmax(pred_logits, axis=-1)  # [N, D, L]

        # Use bins to create the final prediction
        bins = tf.expand_dims(tf.constant(self._bins, dtype=pred_probs.dtype), axis=0)  # [1, D, L]

        pred = tf.reduce_sum(pred_probs * bins, axis=-1)  # [N, D]
        return pred

    @classmethod
    def restore(cls, output_file: str):
        serialized = read_pickle_gz(output_file)

        hidden_units = serialized['W1:0'].shape[-1]
        num_bins = serialized['bins'].shape[-1]

        model = cls(hidden_units=hidden_units, num_bins=num_bins)
        
        model._W1 = serialized['W1:0']
        model._b1 = serialized['b1:0']
        model._W2 = serialized['W2:0']
        model._b2 = serialized['b2:0']

        model._bins = serialized['bins']
        model._bin_endpoints = serialized['bin_endpoints']

        return model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--model-type', type=str, choices=['linear', 'dropout', 'interval', 'bootstrap', 'quantile'], required=True)
    args = parser.parse_args()

    data_file = os.path.join('datasets', args.dataset_name, 'validation', 'data.h5')

    scaler = read_pickle_gz(os.path.join('saved_models', args.dataset_name, 'mlp_scaler.pkl.gz'))
    output_folder = os.path.join('saved_models', args.dataset_name)

    if args.model_type == 'linear':
        transition_model = LinearModel()
    elif args.model_type == 'dropout':
        transition_model = DropoutModel(hidden_units=20)
    elif args.model_type == 'interval':
        transition_model = IntervalModel(hidden_units=16, num_bins=16)
    elif args.model_type == 'bootstrap':
        transition_model = BootstrapModel()
    elif args.model_type == 'quantile':
        transition_model = QuantileModel()
    else:
        raise ValueError('Unknown model type {0}'.format(args.model_type))

    transition_model.train(data_file=data_file,
                           scaler=scaler,
                           output_folder=output_folder)
