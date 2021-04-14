import numpy as np
from typing import List


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


def to_label(label: str) -> str:
    return ' '.join(t.capitalize() for t in label.split('_'))


def geometric_mean(x: List[float]) -> float:
    x_prod = np.prod(x)
    return np.power(x_prod, 1.0 / len(x))
