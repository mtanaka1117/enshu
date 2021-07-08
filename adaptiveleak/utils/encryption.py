from enum import Enum, auto
from Cryptodome.Cipher import AES, ChaCha20
from Cryptodome.Hash import HMAC, SHA256
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.Padding import pad, unpad

from .data_types import EncryptionMode


AES_BLOCK_SIZE = 16
CHACHA_KEY_LEN = 32
CHACHA_NONCE_LEN = 12
SHA256_LEN = 32


def encrypt(message: bytes, key: bytes, mode: EncryptionMode) -> bytes:
    """
    Encrypts the message using the given cipher type.

    Args:
        message: The message to encrypt
        key: The symmetric encryption key
        mode: The encryption mode (block or stream)
    Returns:
        The encrypted message
    """
    if mode == EncryptionMode.BLOCK:
        return encrypt_aes128(message=message, key=key)
    elif mode == EncryptionMode.STREAM:
        return encrypt_chacha20(message=message, key=key)
    else:
        raise ValueError('Unknown encryption mode: {0}'.format(mode))


def decrypt(ciphertext: bytes, key: bytes, mode: EncryptionMode) -> bytes:
    """
    Decrypts the text using the given cipher mode.

    Args:
        ciphertext: The ciphertext to decrypt
        key: The symmetric encryption key
        mode: The encryption mode (block or stream)
    Returns:
        The plaintext message
    """
    if mode == EncryptionMode.BLOCK:
        return decrypt_aes128(ciphertext=ciphertext, key=key)
    elif mode == EncryptionMode.STREAM:
        return decrypt_chacha20(ciphertext=ciphertext, key=key)
    else:
        raise ValueError('Unknown encryption mode: {0}'.format(mode))


def encrypt_aes128(message: bytes, key: bytes) -> bytes:
    """
    Encrypts the given message using an AES-128 block cipher.

    Args:
        message: The message to encrypt
        key: The symmetric encryption key
    Returns:
        The ciphertext
    """
    assert len(key) == AES_BLOCK_SIZE, 'Must provide a {0}-byte key. Got {1} bytes.'.format(AES_BLOCK_SIZE, len(key))

    # Create a random initialization vector
    iv = get_random_bytes(AES_BLOCK_SIZE)

    # Pad the message if needed
    if (len(message) % AES_BLOCK_SIZE) != 0:
        message = pad(message, block_size=AES_BLOCK_SIZE, style='x923')

    # Encrypt the message
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(message)

    # Include the IV in the message
    return iv + ciphertext


def decrypt_aes128(ciphertext: bytes, key: bytes) -> bytes:
    """
    Decrypts the given cipher text using the AES-128 block cipher.

    Args:
        ciphertext: The cipher text to decrypt
        key: The symmetric encryption key
    Returns:
        The plaintext message
    """
    assert len(key) == AES_BLOCK_SIZE, 'Must provide a {0}-byte key. Got {1} bytes.'.format(AES_BLOCK_SIZE, len(key))

    # Extract the IV and message components
    iv = ciphertext[:AES_BLOCK_SIZE]
    ciphertext = ciphertext[AES_BLOCK_SIZE:]

    # Decrypt the message
    cipher = AES.new(key, AES.MODE_CBC, iv)
    message = cipher.decrypt(ciphertext)

    return unpad(message, block_size=AES_BLOCK_SIZE, style='x923')


def encrypt_chacha20(message: bytes, key: bytes) -> bytes:
    """
    Encrypts the message using the ChaCha20 Stream Cipher.

    Args:
        message: The plaintext message to encrypt
        key: The symmetric encryption key
    Returns:
        The corresponding ciphertext
    """
    assert len(key) == CHACHA_KEY_LEN, 'Must provide a {0}-byte key. Got {1} bytes'.format(CHACHA_KEY_LEN, len(key))

    # Create the nonce
    nonce = get_random_bytes(CHACHA_NONCE_LEN)

    # Encrypt the message
    cipher = ChaCha20.new(key=key, nonce=nonce)
    ciphertext = cipher.encrypt(message)

    return nonce + ciphertext


def decrypt_chacha20(ciphertext: bytes, key: bytes) -> bytes:
    """
    Decrypts the given cipher text using the ChaCha20 stream cipher.

    Args:
        ciphertext: The cipher text to decrypt
        key: The symmetric encryption key
    Returns:
        The plaintext message
    """
    assert len(key) == CHACHA_KEY_LEN, 'Must provide a {0}-byte key. Got {1} bytes'.format(CHACHA_KEY_LEN, len(key))

    # Extract the Nonce
    nonce = ciphertext[:CHACHA_NONCE_LEN]
    ciphertext = ciphertext[CHACHA_NONCE_LEN:]

    # Decrypt the message
    cipher = ChaCha20.new(key=key, nonce=nonce)
    message = cipher.decrypt(ciphertext)

    return message


def add_hmac(message: bytes, secret: bytes) -> bytes:
    """
    Generates and appends a message authentication code
    to the given message.

    Args:
        message: The message being sent
        secret: The HMAC secret
    Returns:
        The message with the MAC pre-pended
    """
    hmac = HMAC.new(key=secret, msg=message, digestmod=SHA256)
    return hmac.digest() + message


def verify_hmac(mac: bytes, message: bytes, secret: bytes) -> bool:
    """
    Verifies the MAC-prepended message and strips away the MAC.

    Args:
        mac: The MAC tag.
        message: The received message (without the MAC).
        secret: The HMAC secret
    Returns:
        Whether the MAC is verified or not
    """
    # Verify the MAC
    hmac = HMAC.new(key=secret, msg=message, digestmod=SHA256)

    try:
        hmac.verify(mac)
        return True
    except ValueError:
        return False
