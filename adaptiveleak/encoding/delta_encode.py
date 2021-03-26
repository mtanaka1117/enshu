import numpy as np

class DeltaEncode:

    def __init__(self, raw: np.ndarray):
        assert len(raw.shape) == 1, 'Must pass a 1d array'

        self._raw = raw  # [T - 1]

        self._start = raw[0]
        self._diff = raw[1:] - raw[:-1]  # [T - 1]

    def get_start(self) -> float:
        return self._start

    def get_diffs(self) -> np.ndarray:
        return self._diff

    def decode(self) -> np.ndarray:
        cum_diff = np.cumsum(self._diff)  # [T - 1]
        cum_diff = np.concatenate([[0], cum_diff], axis=0)  # [T]
        return self._start + cum_diff  # [T]
