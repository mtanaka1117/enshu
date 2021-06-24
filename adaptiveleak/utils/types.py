from enum import Enum, auto
from collections import namedtuple

class PolicyType(Enum):
    ADAPTIVE_HEURISTIC = auto()
    ADAPTIVE_DEVIATION = auto()
    SKIP_RNN = auto()
    UNIFORM = auto()
    RANDOM = auto()


class EncryptionMode(Enum):
    BLOCK = auto()
    STREAM = auto()


class EncodingMode(Enum):
    STANDARD = auto()
    GROUP = auto()


PolicyResult = namedtuple('PolicyResult', ['measurements', 'collected_indices', 'encoded', 'energy', 'num_bytes', 'num_collected'])
