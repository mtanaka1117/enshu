import matplotlib.pyplot as plt
import os
import numpy as np
from argparse import ArgumentParser
from scipy import stats
from collections import namedtuple, OrderedDict
from typing import Any, Dict, List, Optional

from utils.file_utils import read_pickle_gz


SimResult = namedtuple('SimResult', ['inference', 'attack', 'targets', 'size', 'name'])
MODEL_ORDER = ['Random', 'Uniform', 'Adaptive', 'Adaptive Block', 'Adaptive Stream']

COLORS = {
    'Random': '#d73027',
    'Uniform': '#fc8d59',
    'Adaptive': '#9ecae1',
    'Adaptive Stream': '#6baed6',
    'Adaptive Block': '#08519c'
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


def geometric_mean(x: List[float]) -> float:
    x_prod = np.prod(x)
    return np.power(x_prod, 1.0 / len(x))


def plot(sim_results: Dict[str, SimResult], dataset_name: str, output_file: Optional[str]):

    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots()

        for name in MODEL_ORDER:
            if name not in sim_results:
                continue

            r = sim_results[name]
            ax.plot(r.targets, r.attack, marker='o', linewidth=4, markersize=8, label=r.name, color=COLORS[r.name])

            print('{0} & {1:.4f}'.format(r.name, np.max(r.attack)))

        ax.set_xlabel('Fraction of Measurements')
        
        ax.set_ylabel('Attack Accuracy')

        ax.set_title('Attacker Accuracy on the {0} Dataset'.format(dataset_name.capitalize()))

        ax.legend()

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file, bbox_inches='tight')
        

def extract_results(folder: str) -> SimResult:

    accuracy: Dict[float, Tuple[float, float]] = dict()

    for file_name in sorted(os.listdir(folder)):
        path = os.path.join(folder, file_name)
        serialized = read_pickle_gz(path)

        target = serialized['policy']['target']
        inference = serialized['accuracy']
        attack = serialized['attack']['test_accuracy']
        size = serialized['avg_bytes']
        
        policy_name = get_name(serialized['policy'], is_padded=serialized.get('is_padded', True))
        accuracy[target] = (inference, attack, size)

    inference_list: List[float] = []
    attack_list: List[float] = []
    target_list: List[float] = []
    size_list: List[float] = []

    for target, (inf, att, size) in sorted(accuracy.items()):
        target_list.append(target)
        inference_list.append(inf)
        attack_list.append(att)
        size_list.append(size)

    return SimResult(inference=inference_list,
                     attack=attack_list,
                     targets=target_list,
                     size=size_list,
                     name=policy_name)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--policy-folders', type=str, nargs='+', required=True)
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    sim_results = {r.name: r for r in map(extract_results, args.policy_folders)}
    plot(sim_results, output_file=args.output_file, dataset_name=args.dataset_name)

