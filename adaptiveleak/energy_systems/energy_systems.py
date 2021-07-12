import os.path
import numpy as np

from adaptiveleak.utils.constants import BT_FRAME_SIZE
from adaptiveleak.utils.data_utils import round_to_block, truncate_to_block
from adaptiveleak.utils.encryption import AES_BLOCK_SIZE
from adaptiveleak.utils.file_utils import iterate_dir, read_json
from adaptiveleak.utils.data_types import PolicyType, EncodingMode, CollectMode, EncryptionMode


GROUP_FACTOR = 2


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

        self._comm_w = weights_dict['comm_w']
        self._comm_b = weights_dict['comm_b']

        self._scale = 0.7
        self._rand = np.random.RandomState(57010)

    def get_energy(self, num_bytes: int, use_noise: bool) -> float:
        """
        Returns the energy (in mJ) associated with sending the given number of bytes.
        """
        # Round the energy to the nearest frame
        num_bytes = truncate_to_block(num_bytes, block_size=BT_FRAME_SIZE)

        # Use the linear model to predict the energy amount
        energy = self._comm_w * num_bytes + self._comm_b

        if use_noise:
            energy = self._rand.normal(loc=energy, scale=self._scale)

        return max(energy, 0.0)


class ActiveEnergy:

    def __init__(self):
        # Get the trace data
        dir_name = os.path.dirname(__file__)
        weights_path = os.path.join(dir_name, '..', 'traces', 'bluetooth', 'model.json')
        weights_dict = read_json(weights_path)

        self._base_w = weights_dict['base_w']
        self._base_b = weights_dict['base_b']

        self._scale = 0.001
        self._rand = np.random.RandomState(3513)

    def get_energy(self, num_bytes: int, period: float, use_noise: bool) -> float:
        """
        Returns the energy required for baseline device activity over the given period when
        sending the given number of bytes.
        """
        # Estimate the baseline power
        baseline_power = self._base_w * num_bytes + self._base_b

        # Compute the baseline energy using the given period
        energy = baseline_power * period

        if use_noise:
            energy = self._rand.normal(loc=energy, scale=self._scale)

        return max(energy, 0.0)


class EncryptionEnergy:

    def __init__(self, encryption_mode: EncryptionMode):
        self._encryption_mode = encryption_mode

        # Get the base directory
        dir_name = os.path.dirname(__file__)
        base = os.path.join(dir_name, '..', 'traces', 'encryption')

        # Load the linear model
        weights_path = os.path.join(base, 'model.json')
        weights_dict = read_json(weights_path)

        self._w = weights_dict['w']
        self._b = weights_dict['b']

        self._scale = 0.001
        self._rand = np.random.RandomState(seed=8753)

    def get_energy(self, num_bytes: int, use_noise: bool) -> float:
        if self._encryption_mode == EncryptionMode.BLOCK:
            num_bytes = round_to_block(num_bytes, block_size=AES_BLOCK_SIZE)

        # Use the linear model to predict the energy amount
        energy = self._w * num_bytes + self._b

        if use_noise:
            energy = self._rand.normal(loc=energy, scale=self._scale)

        return max(energy, 0.0)


class EncodingEnergy:

    def __init__(self, encoding_mode: EncodingMode):
        # As a conservative estimate, we set the group variants
        # to the cost of the standard encoding algorithm
        if encoding_mode in (EncodingMode.GROUP_UNSHIFTED, EncodingMode.SINGLE_GROUP):
            encoding_mode = EncodingMode.STANDARD

        self._encoding_mode = encoding_mode

        # Get the base directory
        dir_name = os.path.dirname(__file__)
        base = os.path.join(dir_name, '..', 'traces', 'encode', encoding_mode.name.lower())

        # Load the linear model
        weights_path = os.path.join(base, 'model.json')
        weights_dict = read_json(weights_path)

        self._w = weights_dict['w']
        self._b = weights_dict['b']

        self._scale = 0.001
        self._rand = np.random.RandomState(seed=52792)

    def get_energy(self, num_features: int, use_noise: bool) -> float:
        # Use the linear model to predict the energy amount
        energy = self._w * num_features + self._b

        if use_noise:
            energy = self._rand.normal(loc=energy, scale=self._scale)

        energy = max(energy, 0.0)

        if self._encoding_mode == EncodingMode.GROUP:
            return energy * GROUP_FACTOR

        return energy


class CollectEnergy:

    def __init__(self, collect_mode: CollectMode):
        if (collect_mode == CollectMode.TINY):
            # Get the path
            dir_name = os.path.dirname(__file__)
            energy_path = os.path.join(dir_name, '..', 'traces', 'collect', 'energy.json')
            energy_dict = read_json(energy_path)

            # Read the energy value
            self._energy = np.median(energy_dict['energy'])
            self._scale = np.std(energy_dict['energy'])
        elif (collect_mode == CollectMode.LOW):
            self._energy = 0.1
            self._scale = 0.001
        elif (collect_mode == CollectMode.MED):
            self._energy = 1.0
            self._scale = 0.01
        elif (collect_mode == CollectMode.HIGH):
            self._energy = 10.0
            self._scale = 0.1
        else:
            raise ValueError('Unknown collection mode: {0}'.format(collect_mode.name))

        self._rand = np.random.RandomState(seed=8753)

    def get_energy(self, use_noise: bool) -> float:
        return self.get_energy_multiple(count=1, use_noise=use_noise)

    def get_energy_multiple(self, count: int, use_noise: bool) -> float:
        if not use_noise:
            return self._energy * count

        noisy_energy = self._rand.normal(loc=self._energy, scale=self._scale, size=count)
        return float(np.sum(np.maximum(noisy_energy, 0.0)))


class PolicyComponentEnergy:

    def __init__(self, name: str, op_name: str, seed: int):
        # Get the base directory
        dir_name = os.path.dirname(__file__)
        energy_path = os.path.join(dir_name, '..', 'traces', op_name, name.lower(), 'energy.json')
        energy_dict = read_json(energy_path)

        # Read the energy values
        self._energy = np.median(energy_dict['energy'])
        self._scale = np.std(energy_dict['energy'])

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


class EnergyUnit:

    def __init__(self,
                 policy_type: PolicyType,
                 encryption_mode: EncryptionMode,
                 encoding_mode: EncodingMode,
                 collect_mode: CollectMode,
                 seq_length: int,
                 num_features: int,
                 period: float):
        """
        Creates the energy unit for a given policy.

        Args:
            policy_type: The type of policy
            encoding_mode: The encoding mode (standard or group)
            encryption_mode: The encryption mode (block or stream)
            collect_mode: The collection mode (low, medium, high)
            seq_length: The length of each sequence
            period: The number of seconds per sequence
        """
        # Make the different energy components
        self._should_collect = ShouldCollectEnergy(policy_type=policy_type)
        self._update = UpdateEnergy(policy_type=policy_type)
        self._encode = EncodingEnergy(encoding_mode=encoding_mode)
        self._encrypt = EncryptionEnergy(encryption_mode=encryption_mode)
        self._collect = CollectEnergy(collect_mode=collect_mode)
        self._comm = BluetoothEnergy()
        self._active = ActiveEnergy()

        # Save the different parameters
        self._collect_mode = collect_mode
        self._encoding_mode = encoding_mode
        self._encryption_mode = encryption_mode

        # Save the sequence length, number of features, and period
        self._seq_length = seq_length
        self._num_features = num_features
        self._period = period

    def get_computation_energy(self, num_bytes: int, num_collected: int, use_noise: bool) -> float:
        # Get the energy from each component
        collect_energy = self._collect.get_energy_multiple(count=num_collected,
                                                           use_noise=use_noise)

        should_collect_energy = self._should_collect.get_energy_multiple(count=(self._seq_length - num_collected),
                                                                         use_noise=use_noise)  # The update accounts for 'should_collect'

        update_energy = self._update.get_energy_multiple(count=num_collected,
                                                         use_noise=use_noise)

        encode_energy = self._encode.get_energy(num_features=self._num_features * num_collected,
                                                use_noise=use_noise)

        encrypt_energy = self._encrypt.get_energy(num_bytes=num_bytes,
                                                  use_noise=use_noise)

        return collect_energy + should_collect_energy + update_energy + encode_energy + encrypt_energy

    def get_communication_energy(self, num_bytes: int, use_noise: bool) -> float:
        return self._comm.get_energy(num_bytes=num_bytes,
                                     use_noise=use_noise)


    def get_active_energy(self, num_bytes: int, use_noise: bool) -> float:
        return self._active.get_energy(num_bytes=num_bytes,
                                       period=self._period,
                                       use_noise=use_noise)

    def get_energy(self, num_collected: int, num_bytes: int, use_noise: bool):
        # Get the energy from each component
        comp_energy = self.get_computation_energy(num_bytes=num_bytes,
                                                  num_collected=num_collected,
                                                  use_noise=use_noise)

        comm_energy = self.get_communication_energy(num_bytes=num_bytes,
                                                    use_noise=use_noise)

        active_energy = self.get_active_energy(num_bytes=num_bytes,
                                               use_noise=use_noise)


        return comp_energy + comm_energy + active_energy

    def __str__(self) -> str:
        return 'Energy Unit -> Collect {0}, Encode: {1}, Encrypt: {2}, Seq Length: {3}, Num Features: {4}, Period: {5}'.format(self._collect_mode, self._encoding_mode, self._encryption_mode, self._seq_length, self._num_features, self._period)


