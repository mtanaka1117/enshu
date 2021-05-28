import numpy as np
import os.path
from itertools import chain
from typing import List, Tuple, Iterable, Dict, Optional, Any

from adaptiveleak.utils.file_utils import iterate_dir, read_json_gz


PLOT_STYLE = 'seaborn-ticks'
MARKER = 'o'
LINE_WIDTH = 3
MARKER_SIZE = 8
LEGEND_FONT = 12
AXIS_FONT = 12
TITLE_FONT = 14
PLOT_SIZE = (8, 6)


COLORS = {
    'random': '#d73027',
    'uniform': '#fc8d59',
    'adaptive_heuristic_standard': '#9ecae1',
    'adaptive_heuristic_group': '#08519c',
    'adaptive_deviation_standard': '#c2a5cf',
    'adaptive_deviation_group': '#7b3294',
    'adaptive_jitter_standard': '#dfc27d',
    'adaptive_jitter_group': '#a6611a'
}

DATASET_NAMES = {
    'strawberry': 'Strawberry',
    'pavement': 'Pavement',
    'epilepsy': 'Epilepsy',
    'uci_har': 'Activity',
    'tiselac': 'Tiselac',
    'haptics': 'Haptics',
    'eog': 'EOG',
    'trajectories': 'Characters'
}


def to_label(label: str) -> str:
    return ' '.join(t.capitalize() for t in label.split('_'))


def dataset_label(dataset: str) -> str:
    return DATASET_NAMES[dataset.lower()]


def geometric_mean(x: List[float]) -> float:
    x_prod = np.prod(x)
    return np.power(x_prod, 1.0 / len(x))


def extract_results(folder: str, field: str, aggregate_mode: Optional[str]) -> Tuple[str, Dict[float, Any]]:

    result: Dict[float, float] = dict()

    for file_name in sorted(os.listdir(folder)):
        path = os.path.join(folder, file_name)
        serialized = read_json_gz(path)

        target = serialized['policy']['target']
        name = serialized['policy']['name']

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

        result[target] = value

    return name, result


def iterate_policy_folders(date_folders: List[str], dataset: str) -> Iterable[str]:
    dataset = dataset.lower()
    return chain(*(iterate_dir(os.path.join('..', 'saved_models', dataset, folder)) for folder in sorted(date_folders)))
