import unittest

from adaptiveleak.policies import AdaptivePolicy, EncodingMode
from adaptiveleak.utils.encryption import EncryptionMode, CHACHA_NONCE_LEN, AES_BLOCK_SIZE
from adaptiveleak.utils.data_utils import round_to_block
from adaptiveleak.utils.file_utils import read_pickle_gz


class TestAdaptiveEncode(unittest.TestCase):

    def test_chlorine_stream(self):
        # Load the data
        sample = read_pickle_gz('chlorine_sample.pkl.gz')

        policy = AdaptivePolicy(target=0.7,
                                threshold=0.0,
                                precision=6,
                                width=8,
                                seq_length=166,
                                num_features=1,
                                window=20,
                                encryption_mode=EncryptionMode.STREAM,
                                encoding_mode=EncodingMode.GROUP)

        expected_bytes = 150

        encoded = policy.encode(measurements=sample['measurements'],
                                collected_indices=sample['indices'])

        encoded_bytes = CHACHA_NONCE_LEN + len(encoded)
        self.assertEqual(expected_bytes, encoded_bytes)

    def test_chlorine_block(self):
        # Load the data
        sample = read_pickle_gz('chlorine_sample.pkl.gz')

        policy = AdaptivePolicy(target=0.7,
                                threshold=0.0,
                                precision=6,
                                width=8,
                                seq_length=166,
                                num_features=1,
                                window=20,
                                encryption_mode=EncryptionMode.BLOCK,
                                encoding_mode=EncodingMode.GROUP)

        expected_bytes = 160

        encoded = policy.encode(measurements=sample['measurements'],
                                collected_indices=sample['indices'])

        encoded_bytes = round_to_block(AES_BLOCK_SIZE + len(encoded), block_size=AES_BLOCK_SIZE)
        self.assertEqual(expected_bytes, encoded_bytes)


if __name__ == '__main__':
    unittest.main()
