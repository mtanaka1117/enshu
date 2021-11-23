# Adaptive Group Encoding Utils
The `utils` module contains many critical functions for Adaptive Group Encoding (AGE). We outline this functionality below.

## Message
The `message.py` file implements message encoding and decoding. The standard process uses the functions `encode_standard_measurements` and `decode_standard_measurements`. The functions `encode_stable_measurements` and `decode_stable_measurements` control the AGE process. Both functions pack features into byte arrays to properly leverage smaller bit widths.

## Shifting
The file `shifting.py` computes the exponent groups for AGE. This process uses a variant of the union-find algorithm to merge consecutive groups with the same exponent. This merging controls the amount of metadata overhead. The code here implements the merging step described in Section 4.3 of the paper.

## Encryption
The file `encryption.py` serves as a wrapper around encryption algorithms from PyCrptodome. The wrapper supports both block (AES) and stream (ChaCha20) ciphers.

## Data Utils
The `data_utils.py` file contains a variety of utility functions using during encoding and sampling. We highlight a few important features below.

1. The functions `to_fixed_point` and `to_float` control quantizing to and from fixed point values.
2. The `select_range_shift` and `select_range_shifts_array` functions compute the exponent shift for each feature value in the given set of measurements. The functions attempt to create long runs of exponents for better compression through run-length encoding. This process implements the exponent computation in Section 4.3 of the paper.
3. The function `set_widths` uses a round-robin algorithm to set the bit width of each group in AGE. This process aims to saturate the given number of target bytes. This function implements the group bit-width setting described in Section 4.4 of the paper.
4. The routine `calculate_bytes` projects the number of bytes required by the standard encoding process. This projection occurs without the overhead of actually creating the message.
5. The function `calculate_grouped_bytes` computes the number of bytes needed by a message encoding by AGE. This computation occurs without creating the final message.
6. The function `prune_sequence` removes measurements from the given array to meet the given maximum number of collected elements. This process follows Section 4.2 in the paper.
