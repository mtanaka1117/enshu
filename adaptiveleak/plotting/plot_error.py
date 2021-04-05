import matplotlib.pyplot as plt
import os
import numpy as np
from argparse import ArgumentParser
from scipy import stats
from collections import namedtuple, OrderedDict
from typing import Any, Dict, List, Optional

from adaptiveleak.utils.file_utils import read_json_gz


SimResult = namedtuple('SimResult', ['inference', 'targets', 'reconstruct', 'name'])
MODEL_ORDER = ['random', 'uniform', 'adaptive_standard', 'adaptive_group']

COLORS = {
    'random': '#d73027',
    'uniform': '#fc8d59',
    'adaptive_standard': '#9ecae1',
    'adaptive_group': '#08519c'
}


def get_name(policy: OrderedDict, is_padded: bool) -> str:
    name = policy['name'].capitalize()

    if name == 'Adaptive':
        compression = policy['compression_name'].capitalize()

        if compression == 'Fixed':
            return 'Adaptive'

        if not is_padded:
            if compression == 'Block':
                return '{0} Stream'.format(name)

        return '{0} {1}'.format(name, compression)

    return name


def to_label(label: str) -> str:
    return ' '.join(t.capitalize() for t in label.split('_'))


def geometric_mean(x: List[float]) -> float:
    x_prod = np.prod(x)
    return np.power(x_prod, 1.0 / len(x))


def plot(sim_results: Dict[str, Dict[float, float]], dataset_name: str, output_file: Optional[str]):

    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots(figsize=(10, 8))

        for name in MODEL_ORDER:
            if name not in sim_results:
                continue

            model_results = sim_results[name]
            targets = list(sorted(model_results.keys()))
            errors = [model_results[t] for t in targets]

            ax.plot(targets, errors, marker='o', linewidth=4, markersize=8, label=to_label(name), color=COLORS[name])

            print('{0} & {1:.4f}'.format(name, np.average(errors)))

        ax.set_xlabel('Fraction of Measurements')
        ax.set_ylabel('Average Reconstruction Error')
        ax.set_title('Average Reconstruction Error on the {0} Dataset'.format(dataset_name.capitalize()))

        ax.legend()

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
        error = np.average(serialized['errors'])
        name = serialized['policy']['name']

        result[target] = error

    return name, result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--policy-folders', type=str, nargs='+', required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    sim_results = {name: res for name, res in map(extract_results, args.policy_folders)}
    plot(sim_results, output_file=args.output_file, dataset_name=args.dataset)

