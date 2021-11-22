import unittest
import numpy as np
from sklearn.metrics import mean_absolute_error

from adaptiveleak.policies import AdaptiveHeuristic, SkipRNN, run_policy, BudgetWrappedPolicy
from adaptiveleak.server import reconstruct_sequence
from adaptiveleak.utils.constants import SMALL_NUMBER, LENGTH_SIZE
from adaptiveleak.utils.encryption import CHACHA_NONCE_LEN, AES_BLOCK_SIZE
from adaptiveleak.utils.data_types import EncodingMode, EncryptionMode, CollectMode
from adaptiveleak.utils.data_utils import round_to_block
from adaptiveleak.utils.file_utils import read_pickle_gz
from adaptiveleak.utils.loading import load_data


class TestAdaptiveEncode(unittest.TestCase):

    def test_chlorine_standard_compressed(self):
         # Load the data
        sample = read_pickle_gz('chlorine_sample.pkl.gz')

        policy = AdaptiveHeuristic(collection_rate=0.7,
                                   threshold=0.0,
                                   precision=5,
                                   width=8,
                                   seq_length=166,
                                   num_features=1,
                                   encryption_mode=EncryptionMode.STREAM,
                                   encoding_mode=EncodingMode.STANDARD,
                                   collect_mode=CollectMode.LOW,
                                   should_compress=True,
                                   max_skip=0,
                                   min_skip=0)
        
        encoded = policy.encode(measurements=sample['measurements'],
                                collected_indices=sample['indices'])

        measurements, indices, _ = policy.decode(encoded)

        error = mean_absolute_error(measurements, sample['measurements'])

        self.assertTrue(error < 0.01)
        self.assertEqual(indices, sample['indices'])

    def test_chlorine_stream(self):
        # Load the data
        sample = read_pickle_gz('chlorine_sample.pkl.gz')

        policy = AdaptiveHeuristic(collection_rate=0.7,
                                   threshold=0.0,
                                   precision=6,
                                   width=8,
                                   seq_length=166,
                                   num_features=1,
                                   encryption_mode=EncryptionMode.STREAM,
                                   encoding_mode=EncodingMode.GROUP,
                                   collect_mode=CollectMode.LOW,
                                   should_compress=False,
                                   max_skip=0,
                                   min_skip=0)
        
        encoded = policy.encode(measurements=sample['measurements'],
                                collected_indices=sample['indices'])

        encoded_bytes = CHACHA_NONCE_LEN + LENGTH_SIZE + len(encoded)
        self.assertEqual(encoded_bytes, 114)

    def test_chlorine_block(self):
        # Load the data
        sample = read_pickle_gz('chlorine_sample.pkl.gz')

        policy = AdaptiveHeuristic(collection_rate=0.7,
                                   threshold=0.0,
                                   precision=6,
                                   width=8,
                                   seq_length=166,
                                   num_features=1,
                                   encryption_mode=EncryptionMode.BLOCK,
                                   encoding_mode=EncodingMode.GROUP,
                                   collect_mode=CollectMode.LOW,
                                   should_compress=False,
                                   max_skip=0,
                                   min_skip=0)

        encoded = policy.encode(measurements=sample['measurements'],
                                collected_indices=sample['indices'])

        encoded_bytes = round_to_block(AES_BLOCK_SIZE + len(encoded), block_size=AES_BLOCK_SIZE)
        self.assertEqual(encoded_bytes, 128)

    def test_activity_stream(self):
        # Read the data
        sample = read_pickle_gz('activity_sample.pkl.gz')

        policy = AdaptiveHeuristic(collection_rate=0.5,
                                   threshold=0.0,
                                   precision=6,
                                   width=8,
                                   seq_length=50,
                                   num_features=6,
                                   encryption_mode=EncryptionMode.STREAM,
                                   encoding_mode=EncodingMode.GROUP,
                                   collect_mode=CollectMode.LOW,
                                   should_compress=False,
                                   max_skip=0,
                                   min_skip=0)

        encoded = policy.encode(measurements=sample['measurements'],
                                collected_indices=sample['indices'])

        encoded_bytes = CHACHA_NONCE_LEN + LENGTH_SIZE + len(encoded)

        # Rounded down leads to 130 bytes
        self.assertEqual(encoded_bytes, 130)

    def test_epilepsy_stream(self):
         # Read the data
        sample = read_pickle_gz('epilepsy_sample.pkl.gz')

        policy = AdaptiveHeuristic(collection_rate=0.3,
                                   threshold=0.0,
                                   precision=6,
                                   width=8,
                                   seq_length=206,
                                   num_features=3,
                                   encryption_mode=EncryptionMode.STREAM,
                                   encoding_mode=EncodingMode.GROUP,
                                   collect_mode=CollectMode.LOW,
                                   should_compress=False,
                                   max_skip=0,
                                   min_skip=0)

        encoded = policy.encode(measurements=sample['measurements'],
                                collected_indices=sample['indices'])

        decoded_measurements, decoded_collected, _ = policy.decode(encoded)

        self.assertEqual(len(decoded_measurements), 72)

    def test_tiselac_stream(self):
         # Read the data
        sample = read_pickle_gz('tiselac_sample.pkl.gz')

        policy = AdaptiveHeuristic(collection_rate=0.9,
                                   threshold=0.0,
                                   precision=0,
                                   width=13,
                                   seq_length=23,
                                   num_features=10,
                                   encryption_mode=EncryptionMode.STREAM,
                                   encoding_mode=EncodingMode.GROUP,
                                   collect_mode=CollectMode.LOW,
                                   should_compress=False,
                                   max_skip=0,
                                   min_skip=0)

        encoded = policy.encode(measurements=sample['measurements'],
                                collected_indices=sample['indices'])

        decoded_measurements, decoded_collected, _ = policy.decode(encoded)

        self.assertTrue(np.all(np.isclose(sample['measurements'], decoded_measurements)))

    def test_haptics_stream(self):
        # Read the data
        sample = read_pickle_gz('haptics_sample.pkl.gz')

        policy = AdaptiveHeuristic(collection_rate=0.2,
                                   threshold=0.0,
                                   precision=5,
                                   width=10,
                                   seq_length=1092,
                                   num_features=1,
                                   encryption_mode=EncryptionMode.STREAM,
                                   encoding_mode=EncodingMode.GROUP,
                                   collect_mode=CollectMode.LOW,
                                   should_compress=False,
                                   max_skip=0,
                                   min_skip=0)

        encoded = policy.encode(measurements=sample['measurements'],
                                collected_indices=sample['indices'])

        self.assertEqual(len(encoded), 372)

    def test_activity_stream_1109(self):
        # Read the data
        sample = read_pickle_gz('uci_har_sample_1109.pkl.gz')

        policy = SkipRNN(collection_rate=0.2,
                         threshold=0.0,
                         precision=13,
                         width=16,
                         seq_length=50,
                         num_features=6,
                         encryption_mode=EncryptionMode.STREAM,
                         encoding_mode=EncodingMode.GROUP,
                         collect_mode=CollectMode.LOW,
                         should_compress=False,
                         dataset_name='uci_har')

        encoded = policy.encode(measurements=sample['measurements'],
                                collected_indices=sample['indices'])

        encoded_bytes = len(encoded) + CHACHA_NONCE_LEN + LENGTH_SIZE

        self.assertEqual(encoded_bytes, 114)  # Round down to lower frame

    def test_strawberry_e2e_56(self):
        # Read the data
        inputs, _ = load_data(dataset_name='strawberry', fold='validation')
        inputs = inputs[56]

        # Make the policy
        policy = BudgetWrappedPolicy(name='adaptive_heuristic',
                                     seq_length=inputs.shape[0],
                                     num_features=inputs.shape[1],
                                     encryption_mode='stream',
                                     encoding='group',
                                     collection_rate=0.5,
                                     collect_mode='low',
                                     dataset='strawberry',
                                     should_compress=False)
        
        # Run the policy (we care about sampling correctness here, not the budget)
        policy.init_for_experiment(num_sequences=1)  
        policy_result = run_policy(policy, sequence=inputs, should_enforce_budget=False)

        # Decode the result
        decoded, collected_indices, _ = policy.decode(message=policy_result.encoded)

        # Compare the decoded sequences to the original sampled sequence
        diff = np.abs(decoded - policy_result.measurements)
        max_idx = np.argmax(diff)

        comparison = np.average(diff)
        self.assertLessEqual(comparison, 1e-3)

        # Reconstruct the sequence
        recovered = reconstruct_sequence(measurements=decoded,
                                         collected_indices=collected_indices,
                                         seq_length=inputs.shape[0])

        # Compute the error
        error = np.average(np.abs(recovered - inputs))

        self.assertLess(error, 0.02)

    def test_eog_e2e_44(self):
        # Read the data
        inputs, _ = load_data(dataset_name='eog', fold='validation')
        inputs = inputs[44]

        # Make the policy
        policy = BudgetWrappedPolicy(name='adaptive_heuristic',
                                     seq_length=inputs.shape[0],
                                     num_features=inputs.shape[1],
                                     encryption_mode='stream',
                                     encoding='group',
                                     collection_rate=0.7,
                                     collect_mode='low',
                                     dataset='eog',
                                     should_compress=False)
        
        # Run the policy (we care about sampling correctness here, not the budget)
        policy.init_for_experiment(num_sequences=1)  
        policy_result = run_policy(policy, sequence=inputs, should_enforce_budget=False)

        # Decode the result
        decoded, collected_indices, _ = policy.decode(message=policy_result.encoded)

        # Compare the decoded sequences to the original sampled sequence
        diff = np.abs(decoded - policy_result.measurements)
        max_idx = np.argmax(diff)

        comparison = np.average(diff)
        self.assertLessEqual(comparison, 0.1)

        # Reconstruct the sequence
        recovered = reconstruct_sequence(measurements=decoded,
                                         collected_indices=collected_indices,
                                         seq_length=inputs.shape[0])

        # Compute the error
        error = np.average(np.abs(recovered - inputs))

        self.assertLess(error, 1.0)


if __name__ == '__main__':
    unittest.main()
