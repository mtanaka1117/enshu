import os.path
import csv
import numpy as np
from sklearn.linear_model import Ridge

from adaptiveleak.utils.file_utils import iterate_dir
from adaptiveleak.utils.types import PolicyType, EncodingMode


BT_FRAME_SIZE = 20
OP_TRIALS = 5
BASELINE_PERIOD = 10


def get_energy_from_trace_file(path: str) -> float:
    with open(path, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for idx, line in enumerate(reader):
            if idx > 0:
                energy = float(line[-1])

    return energy / 1000.0


def get_energy_from_folder(path: str, baseline: float, num_trials: int) -> float:
    """
    Reads the given trace file and returns the median energy after
    subtracting out the baseline amount.
    """
    energy_list: List[float] = []

    for trace_path in iterate_dir(path, pattern='.*csv'):
        energy = get_energy_from_trace_file(path=trace_path)
        energy_per_trial = (energy - baseline) / num_trials
        energy_list.append(energy_per_trial)

    return np.average(energy_list), np.std(energy_list)


class BluetoothEnergy:

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
        baseline_energy, _ = get_energy_from_folder(path=os.path.join(base, 'baseline'), baseline=0.0, num_trials=1)

        # Get the energy for each trace value
        for trace_folder in iterate_dir(base, pattern='.*'):
            name = os.path.split(trace_folder)[-1]
    
            try:
                num_bytes = int(name)
            except ValueError:
                continue

            for path in iterate_dir(trace_folder, '.*.csv'):
                energy = get_energy_from_trace_file(path=path)

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
        num_bytes = int(num_bytes / BT_FRAME_SIZE) * BT_FRAME_SIZE
        
        # Use the linear model to predict the energy amount
        energy = self._model.predict(X=[[num_bytes]])[0]

        return max(self._rand.normal(loc=energy, scale=self._scale), 0.0)


class EncryptionEnergy:

    def __init__(self):
        # Get the base directory
        dir_name = os.path.dirname(__file__)
        base = os.path.join(dir_name, '..', 'traces')

        # Get the baseline energy
        baseline_energy, _ = get_energy_from_folder(path=os.path.join(base, 'no_op'), baseline=0.0, num_trials=1)

        # Get the directory corresponding to this policy
        self._energy, self._scale = get_energy_from_folder(path=os.path.join(base, 'encryption'), baseline=baseline_energy, num_trials=OP_TRIALS)

        self._rand = np.random.RandomState(seed=8753)

    def get_energy(self) -> float:
        return max(self._rand.normal(loc=self._energy, scale=self._scale), 0.0)


class CollectEnergy:

    def __init__(self):
        # Get the base directory
        dir_name = os.path.dirname(__file__)
        base = os.path.join(dir_name, '..', 'traces')

        # Get the baseline energy
        baseline_energy, _ = get_energy_from_folder(path=os.path.join(base, 'no_op'), baseline=0.0, num_trials=1)

        # Get the directory corresponding to this policy
        self._energy, self._scale = get_energy_from_folder(path=os.path.join(base, 'collect'), baseline=baseline_energy, num_trials=OP_TRIALS)

        self._rand = np.random.RandomState(seed=8753)

    def get_energy(self) -> float:
        return self.get_energy_multiple(count=1)

    def get_energy_multiple(self, count: int) -> np.ndarray:
        return float(np.sum(np.maximum(self._rand.normal(loc=self._energy, scale=self._scale, size=count), 0.0)))


class PolicyComponentEnergy:

    def __init__(self, name: str, op_name: str, seed: int):
        # Get the base directory
        dir_name = os.path.dirname(__file__)
        base = os.path.join(dir_name, '..', 'traces')

        # Get the directory corresponding to this policy
        trace_folder = os.path.join(base, op_name, name.lower())

        # Read the energy values
        if os.path.exists(trace_folder):
            baseline_energy, _ = get_energy_from_folder(path=os.path.join(base, 'no_op'), baseline=0.0, num_trials=1)
            self._energy, self._scale = get_energy_from_folder(path=trace_folder, baseline=baseline_energy, num_trials=OP_TRIALS)
        else:
            self._energy = 0.0
            self._scale = 0.0001

        # Create the random state
        self._rand = np.random.RandomState(seed)

    def get_energy(self) -> float:
        """
        Returns the energy (mJ) for a single operation
        """
        return self.get_energy_multiple(count=1)

    def get_energy_multiple(self, count: int) -> float:
        """
        Returns the energy (mJ) for the given number of operations
        """
        noisy_energy = self._rand.normal(loc=self._energy, scale=self._scale, size=count)
        return float(np.sum(np.maximum(noisy_energy, 0.0)))


class UpdateEnergy(PolicyComponentEnergy):

    def __init__(self, policy_type: PolicyType):
        super().__init__(name=policy_type.name.lower(),
                         op_name='update',
                         seed=3089)


class ShouldCollectEnergy(PolicyComponentEnergy):

    def __init__(self, policy_type: PolicyType):
        super().__init__(name=policy_type.name.lower(),
                         op_name='should_collect',
                         seed=92093)


class EncodingEnergy(PolicyComponentEnergy):
    
    def __init__(self, encoding_mode: EncodingMode):
        super().__init__(name=encoding_mode.name.lower(),
                         op_name='encode',
                         seed=52782)


class EnergyUnit:

    def __init__(self, policy_type: PolicyType, encoding_mode: EncodingMode, seq_length: int, period: float):
        """
        Creates the energy unit for a given policy.

        Args:
            policy_type: The type of policy
            encoding_mode: The encoding mode (standard or group)
            seq_length: The length of each sequence
            period: The number of seconds per sequence
        """
        # Make the different energy components
        self._should_collect = ShouldCollectEnergy(policy_type=policy_type)
        self._update = UpdateEnergy(policy_type=policy_type)
        self._encode = EncodingEnergy(encoding_mode=encoding_mode)
        self._encrypt = EncryptionEnergy()
        self._collect = CollectEnergy()
        self._comm = BluetoothEnergy()

        # Save the sequence length and period
        self._seq_length = seq_length
        self._period = period
        self._scale = period / BASELINE_PERIOD

        # Read the baseline energy
        dir_name = os.path.dirname(__file__)
        base = os.path.join(dir_name, '..', 'traces')
        baseline_energy, _ = get_energy_from_folder(path=os.path.join(base, 'baseline'), baseline=0.0, num_trials=1)

        self._baseline_energy = baseline_energy * self._scale

    def get_energy(self, num_collected: int, num_bytes: int):
        # Get the energy from each component
        collect_energy = self._collect.get_energy_multiple(count=num_collected)
        should_collect_energy = self._should_collect.get_energy_multiple(count=self._seq_length)
        update_energy = self._update.get_energy_multiple(count=num_collected)
        encode_energy = self._encode.get_energy()
        encrypt_energy = self._encrypt.get_energy()
        comm_energy = self._comm.get_energy(num_bytes=num_bytes)

        return collect_energy + should_collect_energy + update_energy + encode_energy + \
                encrypt_energy + comm_energy + self._baseline_energy
