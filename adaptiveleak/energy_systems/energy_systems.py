import os.path
import csv
import numpy as np
from sklearn.linear_model import Ridge

from adaptiveleak.utils.file_utils import iterate_dir


FRAME_SIZE = 20


def get_energy(path: str) -> float:
    with open(path, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for idx, line in enumerate(reader):
            if idx > 0:
                energy = float(line[-1])

    return energy / 1000.0


class Bluetooth:

    def __init__(self):
        """
        Initializes the Bluetooth energy tracking module by
        fitting an linear model to trace data.
        """
        # Get the trace data
        dir_name = os.path.dirname(__file__)
        base = os.path.join(dir_name, '..', 'traces', 'bluetooth')

        bytes_list: List[int] = []
        energy_list: List[float] = []

        # Read the baseline energy
        baseline_energy_list: List[float] = []
        for path in iterate_dir(os.path.join(base, 'baseline'), pattern='.*csv'):
            baseline_energy_list.append(get_energy(path=path))

        baseline_energy = np.median(baseline_energy_list)

        # Get the energy for each trace value
        for trace_folder in iterate_dir(base, pattern='.*'):
            name = os.path.split(trace_folder)[-1]

            try:
                num_bytes = int(name)
            except ValueError:
                continue

            for path in iterate_dir(trace_folder, '.*.csv'):
                energy = get_energy(path=path)

                bytes_list.append(num_bytes)
                energy_list.append(energy - baseline_energy)

        self._model = Ridge()
        self._model.fit(X=np.expand_dims(bytes_list, axis=-1),
                        y=energy_list)

        self._scale = 0.7
        self._rand = np.random.RandomState(57010)


    def get_energy(self, num_bytes: int) -> float:
        """
        Returns the energy (in mJ) associated with sending the given number of bytes.
        """
        # Round the energy to the nearest frame
        num_bytes = int(num_bytes / FRAME_SIZE) * FRAME_SIZE
        
        # Use the linear model to predict the energy amount
        energy = self._model.predict(X=[[num_bytes]])[0]

        return energy + self._rand.normal(loc=0.0, scale=self._scale)


bt = Bluetooth()
print(bt.get_energy(680))
