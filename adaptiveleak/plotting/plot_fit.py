import matplotlib.pyplot as plt
import os
import numpy as np
from argparse import ArgumentParser
from scipy import stats
from collections import namedtuple, OrderedDict
from typing import Any, Dict, List, Optional

from adaptiveleak.utils.constants import POLICIES
from adaptiveleak.utils.file_utils import read_json_gz
from adaptiveleak.plotting.plot_utils import COLORS, to_label, geometric_mean, MARKER, MARKER_SIZE, LINE_WIDTH, PLOT_STYLE
from adaptiveleak.plotting.plot_utils import PLOT_SIZE, AXIS_FONT, LEGEND_FONT, TITLE_FONT



def plot(sim_results: Dict[str, Dict[float, float]], dataset_name: str, output_file: Optional[str]):

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=PLOT_SIZE)

        for name in POLICIES:
            if name not in sim_results:
                continue

            model_results = sim_results[name]
            targets = list(sorted(model_results.keys()))
            errors = [model_results[t] for t in targets]

            ax.plot(targets, errors, marker=MARKER, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=to_label(name), color=COLORS[name])

            print('{0} & {1:.4f}'.format(name, np.average(errors)))

        ax.set_xlabel('Fraction of Measurements', fontsize=AXIS_FONT)
        ax.set_ylabel('$R^2$', fontsize=AXIS_FONT)
        ax.set_title('$R^2$ on the {0} Dataset'.format(dataset_name.capitalize()), fontsize=TITLE_FONT)

        ax.legend(fontsize=LEGEND_FONT)

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file, bbox_inches='tight')
        

def extract_results(folder: str) -> Dict[float, float]:

    result: Dict[float, float] = dict()

    for file_name in sorted(os.listdir(folder)):
        path = os.path.join(folder, file_name)
        serialized = read_json_gz(path)

        target = serialized['policy']['target']
        r2 = serialized['r2_score']
        name = serialized['policy']['name']

        result[target] = r2

    return name, result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--policy-folders', type=str, nargs='+', required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    sim_results = {name: res for name, res in map(extract_results, args.policy_folders)}
    plot(sim_results, output_file=args.output_file, dataset_name=args.dataset)

