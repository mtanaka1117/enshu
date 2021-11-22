import numpy as np
import os.path
from itertools import chain
from typing import List, Tuple, Iterable, Dict, Optional, Any

from adaptiveleak.utils.file_utils import iterate_dir, read_json_gz


PLOT_STYLE = 'seaborn-ticks'
MARKER = 'o'
LINE_WIDTH = 3
MARKER_SIZE = 8
ANNOTATE_FONT = 14
LEGEND_FONT = 16
AXIS_FONT = 16
TITLE_FONT = 20
PLOT_SIZE = (8, 6)


COLORS = {
    'random_standard': '#d73027',
    'uniform_standard': '#fc8d59',
    'adaptive_heuristic_standard': '#9ecae1',
    'adaptive_heuristic_group_unshifted': '#6baed6',
    'adaptive_heuristic_single_group': '#6baed6',
    'adaptive_heuristic_padded': '#6baed6',
    'adaptive_heuristic_pruned': '#6baed6',
    'adaptive_heuristic_group': '#08519c',
    'adaptive_deviation_standard': '#c2a5cf',
    'adaptive_deviation_group_unshifted': '#9e9ac8',
    'adaptive_deviation_single_group': '#9e9ac8',
    'adaptive_deviation_padded': '#9e9ac8',
    'adaptive_deviation_pruned': '#9e9ac8',
    'adaptive_deviation_group': '#7b3294',
    'skip_rnn_standard': '#dfc27d',
    'skip_rnn_group': '#a6611a'
}

DATASET_NAMES = {
    'strawberry': 'Strawberry',
    'pavement': 'Pavement',
    'epilepsy': 'Epilepsy',
    'uci_har': 'Activity',
    'tiselac': 'Tiselac',
    'haptics': 'Password',
    'eog': 'EOG',
    'trajectories': 'Characters',
    'mnist': 'MNIST'
}


def to_label(label: str) -> str:
    tokens = ((t.capitalize() if t != 'group' else 'AGE') for t in label.split('_') if t.lower() != 'adaptive')
    return ' '.join((t if t != 'Heuristic' else 'Linear' for t in tokens))


def dataset_label(dataset: str) -> str:
    return DATASET_NAMES[dataset.lower()]


def geometric_mean(x: List[float]) -> float:
    x_prod = np.prod(x)
    return np.power(x_prod, 1.0 / len(x))


def get_multiplier(value: float) -> int:
    if abs(value) > 1:
        return 0

    mult = 10
    power = 1

    while (abs(value) < 1):
        value *= mult
        mult *= 10
        power += 1

    return power - 1


def extract_results(folder: str, field: str, aggregate_mode: Optional[str], default_value: Any = 0.0) -> Tuple[str, Dict[float, Any]]:

    result: Dict[float, float] = dict()

    name = ''

    for path in iterate_dir(folder, '.*json.gz'):
        serialized = read_json_gz(path)

        energy_per_seq = serialized['policy']['energy_per_seq']
        name = '{0}_{1}'.format(serialized['policy']['policy_name'].lower(), serialized['policy']['encoding_mode'].lower())

        if field not in serialized:
            value = default_value
        else:
            if aggregate_mode is None:
                value = serialized[field]
            elif aggregate_mode == 'avg':
                value = np.average(serialized[field])
            elif aggregate_mode == 'median':
                value = np.median(serialized[field])
            elif aggregate_mode == 'max':
                value = np.max(serialized[field])
            elif aggregate_mode == 'geom':
                value = geometric_mean(serialized[field])
            else:
                raise ValueError('Unknown aggregation mode: {0}'.format(aggregate_mode))

        result[energy_per_seq] = value

    return name, result


def iterate_policy_folders(date_folders: List[str], dataset: str) -> Iterable[str]:
    dataset = dataset.lower()
    return chain(*(iterate_dir(os.path.join('..', 'saved_models', dataset, folder)) for folder in sorted(date_folders)))
