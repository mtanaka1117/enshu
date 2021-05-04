import unittest
import numpy as np

from adaptiveleak.policies import AdaptiveHeuristic, EncodingMode
from adaptiveleak.utils.encryption import EncryptionMode, CHACHA_NONCE_LEN, AES_BLOCK_SIZE
from adaptiveleak.utils.data_utils import round_to_block
from adaptiveleak.utils.file_utils import read_pickle_gz


class TestAdaptiveEncode(unittest.TestCase):

    def test_chlorine_standard_compressed(self):
         # Load the data
        sample = read_pickle_gz('chlorine_sample.pkl.gz')

        policy = AdaptiveHeuristic(target=0.7,
                                   threshold=0.0,
                                   precision=6,
                                   width=8,
                                   seq_length=166,
                                   num_features=1,
                                   encryption_mode=EncryptionMode.STREAM,
                                   encoding_mode=EncodingMode.STANDARD,
                                   should_compress=True)
        
        encoded = policy.encode(measurements=sample['measurements'],
                                collected_indices=sample['indices'])

        measurements, indices, _ = policy.decode(encoded)

        self.assertTrue(np.all(np.abs(measurements - sample['measurements']) < 0.01))
        self.assertEqual(indices, sample['indices'])

    def test_chlorine_stream(self):
        # Load the data
        sample = read_pickle_gz('chlorine_sample.pkl.gz')

        policy = AdaptiveHeuristic(target=0.7,
                                   threshold=0.0,
                                   precision=6,
                                   width=8,
                                   seq_length=166,
                                   num_features=1,
                                   encryption_mode=EncryptionMode.STREAM,
                                   encoding_mode=EncodingMode.GROUP,
                                   should_compress=False)
        
        encoded = policy.encode(measurements=sample['measurements'],
                                collected_indices=sample['indices'])

        encoded_bytes = CHACHA_NONCE_LEN + len(encoded)
        self.assertEqual(encoded_bytes, 149)

    def test_chlorine_block(self):
        # Load the data
        sample = read_pickle_gz('chlorine_sample.pkl.gz')

        policy = AdaptiveHeuristic(target=0.7,
                                   threshold=0.0,
                                   precision=6,
                                   width=8,
                                   seq_length=166,
                                   num_features=1,
                                   encryption_mode=EncryptionMode.BLOCK,
                                   encoding_mode=EncodingMode.GROUP,
                                   should_compress=False)

        encoded = policy.encode(measurements=sample['measurements'],
                                collected_indices=sample['indices'])

        encoded_bytes = round_to_block(AES_BLOCK_SIZE + len(encoded), block_size=AES_BLOCK_SIZE)
        self.assertEqual(encoded_bytes, 160)

    def test_activity_stream(self):
        # Read the data
        sample = read_pickle_gz('activity_sample.pkl.gz')

        policy = AdaptiveHeuristic(target=0.5,
                                   threshold=0.0,
                                   precision=6,
                                   width=8,
                                   seq_length=50,
                                   num_features=6,
                                   encryption_mode=EncryptionMode.STREAM,
                                   encoding_mode=EncodingMode.GROUP,
                                   should_compress=False)

        encoded = policy.encode(measurements=sample['measurements'],
                                collected_indices=sample['indices'])

        encoded_bytes = CHACHA_NONCE_LEN + len(encoded)
        self.assertEqual(encoded_bytes, 169)

    def test_epilepsy_stream(self):
         # Read the data
        sample = read_pickle_gz('epilepsy_sample.pkl.gz')

        policy = AdaptiveHeuristic(target=0.3,
                                   threshold=0.0,
                                   precision=6,
                                   width=8,
                                   seq_length=206,
                                   num_features=3,
                                   encryption_mode=EncryptionMode.STREAM,
                                   encoding_mode=EncodingMode.GROUP,
                                   should_compress=False)

        encoded = policy.encode(measurements=sample['measurements'],
                                collected_indices=sample['indices'])

        decoded_measurements, decoded_collected, _ = policy.decode(encoded)

        self.assertEqual(len(decoded_measurements), 117)

    def test_tiselac_stream(self):
         # Read the data
        sample = read_pickle_gz('tiselac_sample.pkl.gz')

        policy = AdaptiveHeuristic(target=0.9,
                                   threshold=0.0,
                                   precision=0,
                                   width=13,
                                   seq_length=23,
                                   num_features=10,
                                   encryption_mode=EncryptionMode.STREAM,
                                   encoding_mode=EncodingMode.GROUP,
                                   should_compress=False)

        encoded = policy.encode(measurements=sample['measurements'],
                                collected_indices=sample['indices'])

        decoded_measurements, decoded_collected, _ = policy.decode(encoded)

        self.assertTrue(np.all(np.isclose(sample['measurements'], decoded_measurements)))


if __name__ == '__main__':
    unittest.main()
