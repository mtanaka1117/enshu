import os.path
import csv
import numpy as np
from sklearn.linear_model import Ridge

from adaptiveleak.utils.file_utils import iterate_dir, read_json
from adaptiveleak.utils.types import PolicyType, EncodingMode, CollectMode


BT_FRAME_SIZE = 20
OP_TRIALS = 5
BASELINE_PERIOD = 10


def get_energy_from_folder(path: str, baseline: float, num_trials: int) -> float:
    """
    Reads the given trace file and returns the median energy after
    subtracting out the baseline amount.
    """
    energy_list: List[float] = []

    trace_path = os.path.join(path, 'energy.json')
    energy_list = read_json(trace_path)['energy']
    energy_list = [(e - baseline) / num_trials for e in energy_list]

    return np.average(energy_list), np.std(energy_list)


class BluetoothEnergy:

    def __init__(self):
        """
        Initializes the Bluetooth energy tracking module by
        fitting an linear model to trace data.
        """
        # Get the trace data
        dir_name = os.path.dirname(__file__)
        weights_path = os.path.join(dir_name, '..', 'traces', 'bluetooth', 'model.json')
        weights_dict = read_json(weights_path)

        self._w = weights_dict['w']
        self._b = weights_dict['b']

        self._scale = 0.7
        self._rand = np.random.RandomState(57010)

    def get_energy(self, num_bytes: int, use_noise: bool) -> float:
        """
        Returns the energy (in mJ) associated with sending the given number of bytes.
        """
        # Round the energy to the nearest frame
        num_bytes = int(num_bytes / BT_FRAME_SIZE) * BT_FRAME_SIZE
        
        # Use the linear model to predict the energy amount
        energy = self._w * num_bytes + self._b

        if use_noise:
            energy = self._rand.normal(loc=energy, scale=self._scale)

        return max(energy, 0.0)


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

    def get_energy(self, use_noise: bool) -> float:
        energy = self._rand.normal(loc=self._energy, scale=self._scale) if use_noise else self._energy
        return max(energy, 0.0)


class CollectEnergy:

    def __init__(self, collect_mode: CollectMode):
        # Get the base directory
        dir_name = os.path.dirname(__file__)
        base = os.path.join(dir_name, '..', 'traces')

        # Get the baseline energy
        baseline_energy, _ = get_energy_from_folder(path=os.path.join(base, 'no_op'), baseline=0.0, num_trials=1)

        # Get the directory corresponding to this policy
        self._energy, self._scale = get_energy_from_folder(path=os.path.join(base, 'collect'), baseline=baseline_energy, num_trials=OP_TRIALS)

        self._rand = np.random.RandomState(seed=8753)

    def get_energy(self, use_noise: bool) -> float:
        return self.get_energy_multiple(count=1, use_noise=use_noise)

    def get_energy_multiple(self, count: int, use_noise: bool) -> float:
        if not use_noise:
            return self._energy * count

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

    def get_energy(self, use_noise: bool) -> float:
        """
        Returns the energy (mJ) for a single operation
        """
        return self.get_energy_multiple(count=1, use_noise=use_noise)

    def get_energy_multiple(self, count: int, use_noise: bool) -> float:
        """
        Returns the energy (mJ) for the given number of operations
        """
        if not use_noise:
            return self._energy * count

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

    def __init__(self, policy_type: PolicyType, encoding_mode: EncodingMode, collect_mode: CollectMode, seq_length: int, period: float):
        """
        Creates the energy unit for a given policy.

        Args:
            policy_type: The type of policy
            encoding_mode: The encoding mode (standard or group)
            collect_mode: The collection mode (low, medium, high)
            seq_length: The length of each sequence
            period: The number of seconds per sequence
        """
        # Make the different energy components
        self._should_collect = ShouldCollectEnergy(policy_type=policy_type)
        self._update = UpdateEnergy(policy_type=policy_type)
        self._encode = EncodingEnergy(encoding_mode=encoding_mode)
        self._encrypt = EncryptionEnergy()
        self._collect = CollectEnergy(collect_mode=collect_mode)
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

    def get_energy(self, num_collected: int, num_bytes: int, use_noise: bool):
        # Get the energy from each component
        collect_energy = self._collect.get_energy_multiple(count=num_collected, use_noise=use_noise)
        should_collect_energy = self._should_collect.get_energy_multiple(count=self._seq_length, use_noise=use_noise)
        update_energy = self._update.get_energy_multiple(count=num_collected, use_noise=use_noise)
        encode_energy = self._encode.get_energy(use_noise=use_noise)
        encrypt_energy = self._encrypt.get_energy(use_noise=use_noise)
        comm_energy = self._comm.get_energy(num_bytes=num_bytes, use_noise=use_noise)

        return collect_energy + should_collect_energy + update_energy + encode_energy + \
                encrypt_energy + comm_energy + self._baseline_energy
