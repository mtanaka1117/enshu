import os.path
import tensorflow as tf
import numpy as np
import sklearn.metrics as metrics
import time
import math
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional, List, Tuple, Set, Union

from adaptiveleak.utils.constants import BIG_NUMBER
from adaptiveleak.utils.file_utils import save_pickle_gz, read_pickle_gz, make_dir, save_json_gz


DEFAULT_HYPERS = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'gradient_clip': 1,
    'num_epochs': 15,
    'patience': 10
}


INPUTS = 'inputs'
INPUT_SHAPE = 'input_shape'
SCALER = 'scaler'
OPTIMIZER_OP = 'optimizer'
PREDICTION_OP = 'prediction'
LOSS_OP = 'loss'

TEST_LOG_FMT = '{0}_test-log.json.gz'
TRAIN_LOG_FMT = '{0}_train-log.json.gz'
MODEL_FILE_FMT = '{0}.pkl.gz'


class NeuralNetwork:

    def __init__(self, name: str, hypers: Dict[str, Any]):
        self._name = name
        self._sess = tf.compat.v1.Session(graph=tf.Graph())
        self._ops: Dict[str, tf.Tensor] = dict()
        self._placeholders: Dict[str, tf.placeholder] = dict()
        self._metadata: Dict[str, Any] = dict()
        self._is_made = False

        # Set the random seed to get reproducible results
        with self._sess.graph.as_default():
            tf.random.set_seed(8547)

        self._rand = np.random.RandomState(seed=4929)

        # Overwrite default hyper-parameters
        self._hypers = {key: val for key, val in DEFAULT_HYPERS.items()}
        self._hypers.update(**hypers)

    @property
    def name(self) -> str:
        return self._name

    @property
    def learning_rate(self) -> float:
        return float(self._hypers['learning_rate'])

    @property
    def learning_rate_decay(self) -> float:
        return float(self._hypers['learning_rate_decay'])

    @property
    def batch_size(self) -> int:
        return int(self._hypers['batch_size'])

    @property
    def gradient_clip(self) -> float:
        return float(self._hypers['gradient_clip'])

    @property
    def num_epochs(self) -> int:
        return int(self._hypers['num_epochs'])

    @property
    def patience(self) -> int:
        return int(self._hypers['patience'])

    @property
    def optimizer_name(self) -> str:
        return str(self._hypers['optimizer'])

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self._metadata['input_shape']

    def get_trainable_vars(self) -> List[tf.Variable]:
        return list(self._sess.graph.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES))

    def count_parameters(self) -> int:
        return sum((np.prod(var.get_shape()) for var in self.get_trainable_vars()))

    def make_graph(self, is_train: bool):
        """
        Creates the computational graph.
        """
        raise NotImplementedError()

    def make_loss(self):
        """
        Creates the loss function. This should give a value
        to self._ops[LOSS_OP].
        """
        raise NotImplementedError()

    def batch_to_feed_dict(self, features: np.ndarray, is_train: bool) -> Dict[tf.compat.v1.placeholder, np.ndarray]:
        # Normalize the inputs
        feature_shape = features.shape
        flattened = features.reshape(-1, feature_shape[-1])
        scaled = self._metadata[SCALER].transform(flattened)
        scaled_features = scaled.reshape(feature_shape)

        return {
            self._placeholders[INPUTS]: scaled_features
        }

    def make_placeholders(self):
        """
        Creates the placeholders for this model.
        """
        input_shape = (None, ) + self.input_shape
        self._placeholders[INPUTS] = tf.compat.v1.placeholder(shape=input_shape,
                                                              dtype=tf.float32,
                                                              name=INPUTS)

    def load_metadata(self, inputs: np.ndarray):
        """
        Loads metadata.
        """
        # Build the input normalization object
        scaler = StandardScaler()

        flattened = inputs.reshape(-1, inputs.shape[-1])
        scaler.fit(flattened)

        # Save results in the meta-data dict
        self._metadata[INPUT_SHAPE] = inputs.shape[1:]
        self._metadata[SCALER] = scaler

    def make(self, is_train: bool):
        """
        Builds the model.
        """
        if self._is_made:
            return

        with self._sess.graph.as_default():
            self.make_placeholders()

            self.make_graph(is_train=is_train)
            assert PREDICTION_OP in self._ops, 'Must create a prediction operation'

            self.make_loss()
            assert LOSS_OP in self._ops, 'Must create a loss operation'

            self._optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

            self.make_training_step()
            assert OPTIMIZER_OP in self._ops, 'Must create an optimizer operation'

        self._is_made = True

    def make_training_step(self):
        """
        Creates the training step for gradient descent
        """
        trainable_vars = self.get_trainable_vars()

        # Compute the gradients
        gradients = tf.gradients(self._ops[LOSS_OP], trainable_vars)

        # Clip Gradients
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip)

        # Prune None values from the set of gradients and apply gradient weights
        pruned_gradients = [(grad, var) for grad, var in zip(clipped_gradients, trainable_vars) if grad is not None]

        # Apply clipped gradients
        optimizer_op = self._optimizer.apply_gradients(pruned_gradients)

        # Add optimization to the operations dict
        self._ops[OPTIMIZER_OP] = optimizer_op

    def init(self):
        """
        Initializes the trainable variables.
        """
        with self._sess.graph.as_default():
            init_op = tf.compat.v1.global_variables_initializer()
            self._sess.run(init_op)

    def execute(self, ops: Union[List[str], str], feed_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Executes the given operations.

        Args:
            ops: Names of operations to execute
            feed_dict: Dictionary supplying values to placeholders
        Returns:
            A dictionary of op_name -> op_result
        """
        assert self._is_made, 'Must call make() first'

        # Turn operations into a list
        ops_list = ops
        if not isinstance(ops, list):
            ops_list = [ops]

        with self._sess.graph.as_default():
            ops_to_run = {op_name: self._ops[op_name] for op_name in ops_list}
            results = self._sess.run(ops_to_run, feed_dict=feed_dict)

        return results

    def test(self, test_inputs: np.ndarray, batch_size: Optional[int]) -> Dict[str, Any]:
        """
        Executes the model on the test fold of the given dataset.
        """
        # Execute the model on the testing samples
        test_batch_size = batch_size if batch_size is not None else self.batch_size

        pred_list: List[np.ndarray] = []  # Record predictions
        expected_list: List[np.ndarray] = []
        loss_sum = 0.0
        num_samples = 0
        num_updates = 0

        start_time = datetime.now()
        test_exec_time = 0.0
        exec_batches = 0.0

        test_ops = [PREDICTION_OP, LOSS_OP, 'skip_gates']

        for batch_idx in range(len(test_inputs)):
            start, end = batch_idx, batch_idx + test_batch_size
            batch_features = test_inputs[start:end]

            feed_dict = self.batch_to_feed_dict(batch_features, is_train=False)

            batch_start = time.perf_counter()
            batch_result = self.execute(ops=test_ops, feed_dict=feed_dict)
            batch_end = time.perf_counter()

            if batch_idx > 0:
                test_exec_time += (batch_end - batch_start)
                exec_batches += 1

            predictions = batch_result[PREDICTION_OP]  # [B, T, D]
            num_updates += np.sum(batch_result['skip_gates'])
            
            pred_list.append(predictions)
            expected_list.append(batch_features)

            loss_sum += batch_result[LOSS_OP] * len(batch_features)
            num_samples += len(batch_features)

        end_time = datetime.now()

        # Un-normalize the predictions
        preds = np.vstack(pred_list)  # [M, T, D]
        expected = np.vstack(expected_list)  # [M, T, D]
        
        preds_shape = preds.shape
        preds = self._metadata[SCALER].inverse_transform(preds.reshape(-1, preds_shape[-1]))  # [M * T, D]
        preds = preds.reshape(preds_shape)  # [M, T, D]

        print('Exec Batches: {0}, Total Time: {1}'.format(exec_batches, test_exec_time))

        # Compute the testing metrics
        mae = np.average(np.abs(preds - expected))
        loss = loss_sum / num_samples
        avg_updates = num_updates / (num_samples * self.input_shape[0])
        time_per_batch = test_exec_time / max(exec_batches, 1.0)

        return {
            'mae': float(mae),
            'loss': float(loss),
            'avg_updates': float(avg_updates),
            'duration': str(end_time - start_time),
            'start_time': start_time.strftime('%Y-%m-%d-%H-%M-%S'),
            'end_time': end_time.strftime('%Y-%m-%d-%H-%M-%S'),
            'time_per_batch': time_per_batch
        }

    def train(self, dataset_name: str, train_inputs: np.ndarray, val_inputs: np.ndarray, save_folder: str, should_print: bool) -> Tuple[str, str]:
        """
        Trains the neural network on the given dataset.
        """
        # Load the meta-data from the train data
        self.load_metadata(inputs=train_inputs)

        # Build the model
        self.make(is_train=True)

        # Initialize the variables
        self.init()

        # Create a name for this training run based on the current date and time
        start_time = datetime.now()
        model_name = self.name

        # Create lists to track the training and validation metrics
        train_loss: List[float] = []
        train_accuracy: List[float] = []
        val_loss: List[float] = []
        val_accuracy: List[float] = []

        # Track best model and early stopping
        best_val_loss = BIG_NUMBER
        num_not_improved = 0

        if should_print:
            num_params = self.count_parameters()
            print('Training model with {0} parameters.'.format(num_params))

        # Augment the save directory with the data-set name and model name
        # for clarity
        make_dir(save_folder)

        save_folder = os.path.join(save_folder, dataset_name)
        make_dir(save_folder)

        train_time = 0.0

        # Create the sample indices for batch creation
        train_idx = np.arange(len(train_inputs))
        num_train_batches = int(math.ceil(train_inputs.shape[0] / self.batch_size))
        num_val_batches = int(math.ceil(val_inputs.shape[0] / self.batch_size))

        for epoch in range(self.num_epochs):

            if should_print:
                print('==========')
                print('Epoch: {0}/{1}'.format(epoch + 1, self.num_epochs))
                print('==========')

            # Execute the training operations
            epoch_train_loss = 0.0
            num_train_samples = 0

            # Shuffle the training indices
            self._rand.shuffle(train_idx)

            train_start = time.time()

            train_ops = [LOSS_OP, OPTIMIZER_OP]
            for batch_idx in range(num_train_batches):
                start, end = batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size
                
                indices = train_idx[start:end]
                batch_features = train_inputs[indices]

                feed_dict = self.batch_to_feed_dict(batch_features, is_train=True)

                start_batch_time = time.perf_counter()
                train_batch_results = self.execute(feed_dict=feed_dict, ops=train_ops)
                end_batch_time = time.perf_counter()

                # Add to the total time for training steps
                train_time += end_batch_time - start_batch_time

                batch_samples = len(batch_features)

                # Add to the running loss. The returned loss is the average over the batch
                # so we multiply by the batch size to get the total loss.
                epoch_train_loss += train_batch_results[LOSS_OP] * batch_samples
                num_train_samples += batch_samples

                if should_print:
                    train_loss_so_far = epoch_train_loss / num_train_samples
                    print('Train Batch {0} / {1}: Loss -> {2:.5f}'.format(batch_idx + 1, num_train_batches, train_loss_so_far), end='\r')

            # Clear the line after the epoch
            if should_print:
                print()

            # Log the train loss
            train_loss.append(epoch_train_loss / num_train_samples)

            train_end = time.time()
            train_batch_time = (train_end - train_start) / batch_idx

            # Execute the validation operations
            epoch_val_loss = 0.0
            num_val_samples = 0

            val_start = time.time()

            val_ops = [LOSS_OP]
            for batch_idx in range(num_val_batches):
                start, end = batch_idx * self.batch_size, (batch_idx + 1 ) * self.batch_size
                batch_features = val_inputs[start:end]

                feed_dict = self.batch_to_feed_dict(batch_features, is_train=False)
                val_batch_results = self.execute(feed_dict=feed_dict, ops=val_ops)

                batch_samples = len(batch_features)

                epoch_val_loss += val_batch_results[LOSS_OP] * batch_samples
                num_val_samples += batch_samples

                if should_print:
                    val_loss_so_far = epoch_val_loss / num_val_samples
                    print('Validation Batch {0} / {1}: Loss -> {2:.5f}'.format(batch_idx + 1, num_val_batches, val_loss_so_far), end='\r')

            if should_print:
                print()

            # Log the validation results
            avg_val_loss = epoch_val_loss / num_val_samples
            val_loss.append(avg_val_loss)

            val_end = time.time()
            val_batch_time = (val_end - val_start) / batch_idx

            # Check if we see improved validation performance
            should_save = False
            if avg_val_loss < best_val_loss:
                should_save = True
                num_not_improved = 0
                best_val_loss = avg_val_loss
            else:
                num_not_improved += 1

            has_ended = bool(epoch == self.num_epochs - 1)
            if num_not_improved >= self.patience:
                if should_print:
                    print('Exiting due to early stopping.')

                has_ended = True

            # Save model if specified
            if should_save:
                if should_print:
                    print('Saving...')
                self.save(save_folder=save_folder, model_name=model_name)

            if has_ended:
                break

        # Log the training results
        end_time = datetime.now()

        train_log = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'duration': str(end_time - start_time),  # Total time (accounting for validation and saving)
            'start_time': start_time.strftime('%Y-%m-%d-%H-%M-%S'),
            'end_time': end_time.strftime('%Y-%m-%d-%H-%M-%S'),
            'train_time': train_time  # Actual training execution time in seconds
        }

        train_log_path = os.path.join(save_folder, TRAIN_LOG_FMT.format(model_name))
        save_json_gz(train_log, train_log_path)

        return save_folder, model_name

    def save(self, save_folder: str, model_name: str):
        """
        Saves the model parameters and associated training parameters
        in the given folder.
        """
        make_dir(save_folder)

        model_file = os.path.join(save_folder, MODEL_FILE_FMT.format(model_name))

        save_dict: Dict[str, Any] = {
            'hypers': self._hypers,
            'metadata': self._metadata
        }

        # Get the trainable variables
        with self._sess.graph.as_default():
            trainable_vars = self._sess.run({var.name: var for var in self.get_trainable_vars()})
            save_dict['trainable_vars'] = trainable_vars
        
        # Save the results
        save_pickle_gz(save_dict, model_file)

    @classmethod
    def restore(cls, model_file: str):
        # Extract the model name
        model_name = model_file.split(os.sep)[1].split('.')[0]

        # Read the saved information
        serialized = read_pickle_gz(model_file)

        # Unpack the hyper-parameters and meta-data
        hypers = serialized['hypers']
        metadata = serialized['metadata']

        # Intialize the new model
        name = model_name.split('-')[0]
        network = cls(name=name, hypers=hypers)

        # Set the meta-data
        network._metadata = metadata

        # Build the model
        network.make(is_train=False)

        # Initialize the model
        network.init()

        # Restore the trainable variables
        with network._sess.graph.as_default():
            saved_vars = serialized['trainable_vars']

            trainable_vars = network.get_trainable_vars()

            assign_ops = []
            for var in trainable_vars:
                var_name = var.name

                if var_name not in saved_vars:
                    print('WARNING: No value for {0}'.format(var_name))
                else:
                    assign_ops.append(tf.compat.v1.assign(var, saved_vars[var_name]))

            network._sess.run(assign_ops)

        return network
