import numpy as np
import os.path
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List

from adaptiveleak.classifiers import BaseClassifier, restore_model
from adaptiveleak.transition_model import TransitionModel
from adaptiveleak.utils.file_utils import read_pickle_gz


class Server:
    """
    This class mimics a server that infers 'missing' objects
    and performs inference
    """
    def __init__(self, transition_model: TransitionModel, inference_path: str, seq_length: int):
        self._transition_model = transition_model
        self._seq_length = seq_length  # T

        #inference_model = read_pickle_gz(inference_path)
        # self._clf: MLPClassifier = inference_model['model']
        # self._scaler: StandardScaler = inference_model['scaler']

        model_name = 'mlp'

        self._clf: BaseClassifier = restore_model(name=model_name, save_folder=inference_path)
        self._scaler: StandardScaler = read_pickle_gz(os.path.join(inference_path, '{0}_scaler.pkl.gz'.format(model_name)))

    @property
    def seq_length(self) -> int:
        return self._seq_length

    def scale(self, measurements: np.ndarray) -> np.ndarray:
        """
        Normalizes the given measurements using the internal scaler.

        Args:
            measurements: A [K, D] array of measurements
        Returns:
            A [K, D] array of scaled measurements
        """
        return self._scaler.transform(measurements)

    def inverse_scale(self, scaled: np.ndarray) -> np.ndarray:
        """
        Un-normalizes the given measurements using the internal scaler.

        Args:
            scaled: A [K, D] array of scaled values
        Returns:
            A [K, D] array of un-normalized values
        """
        return self._scaler.inverse_transform(scaled)

    def recieve(self, recv: np.ndarray, indices: List[int]) -> np.ndarray:
        """
        Collects the given sequence by inferring the missing elements

        Args:
            recv: A [K, D] array of received measurements. The value K must
                be at most the larger sequence length T
            indices: A list of [K] indices of the measurements in the larger
                sequence
        Returns:
            A [T, D] array holding the full sequence with inferred missing elements
        """
        assert recv.shape[0] == len(indices), 'Misaligned measurements ({0}) and indices ({1})'.format(recv.shape, len(indices))
        assert recv.shape[0] <= self.seq_length, 'Can have at most {0} measurements'.format(self.seq_length)

        m = np.zeros(shape=(recv.shape[1], 1))  # [D, 1]
        sent_counter = 0

        recovered_list: List[np.ndarray] = []
        for seq_idx in range(self._seq_length):

            if (sent_counter < len(indices)) and (seq_idx == indices[sent_counter]):
                # The server receives the value from the sensor
                m = recv[sent_counter].reshape(-1, 1)  # [D, 1]
                sent_counter += 1
            else:
                pass
                # The server must infer the value using
                # the transition matrix
                # m = self._transition_model.predict(m)

            recovered_list.append(m.reshape(1, -1))

        return np.vstack(recovered_list)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Preforms inference on the given inputs

        Args:
            inputs: A [N, T, D] array of input features
        Returns:
            A [N] array containing the classification predictions
        """
        # Unpack the shape
        # num_samples, seq_length, num_features = inputs.shape

        # Scale the inputs
        # scaled = self.scale(inputs.reshape(-1, num_features))  # [N * T, D]
        # model_inputs = inputs.reshape(num_samples, -1)  # [N, T * D]

        return self._clf.predict(inputs)  # [N]
