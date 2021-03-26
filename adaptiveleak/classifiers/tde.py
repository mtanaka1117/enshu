import numpy as np
from sktime.classification.all import TemporalDictionaryEnsemble

from .base import BaseClassifier


MAX_CLASSIFIERS = 10


class TemporalDictionary(BaseClassifier):

    @property
    def name(self) -> str:
        return 'tde'

    def fit(self, inputs: np.ndarray, labels: np.ndarray, save_folder: str):
        ndims = len(inputs.shape)
        assert (ndims == 3 and inputs.shape[-1] == 1) or (ndims == 2), 'Must be a univariate time series'

        if ndims == 2:
            inputs = np.expand_dims(inputs, axis=-1)  # [N, T, 1]

        inputs = np.transpose(inputs, axes=[0, 2, 1])
        self._model = TemporalDictionaryEnsemble(max_ensemble_size=MAX_CLASSIFIERS, random_state=2952)

        self._model.fit(inputs, labels)

    def predict_probs(self, inputs: np.ndarray) -> np.ndarray:        
        ndims = len(inputs.shape)
        assert (ndims == 3 and inputs.shape[-1] == 1) or (ndims == 2), 'Must be a univariate time series'

        if ndims == 2:
            inputs = np.expand_dims(inputs, axis=-1)  # [N, T, 1]

        inputs = np.transpose(inputs, axes=[0, 2, 1])
        return self._model.predict_proba(inputs)
