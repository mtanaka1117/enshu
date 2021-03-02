import matplotlib.pyplot as plt
import os
import numpy as np
from argparse import ArgumentParser
from scipy import stats
from collections import namedtuple
from typing import Any, Dict, List, Optional

from utils.file_utils import read_pickle_gz


SimResult = namedtuple('SimResult', ['inference', 'attack', 'targets', 'size', 'name'])

COLORS = {
    'random': '#ca0020',
    'adaptive block': '#f4a582',
    'adaptive fixed': '#92c5de',
    'adaptive pid': '#0571b0'
}


def to_label(label: str) -> str:
    return ' '.join(t.capitalize() for t in label.split())


def geometric_mean(x: List[float]) -> float:
    x_prod = np.prod(x)
    return np.power(x_prod, 1.0 / len(x))


def plot(sim_results: List[SimResult], output_file: Optional[str]):

    with plt.style.context('seaborn-ticks'):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))

        for r in sim_results:
            ax1.plot(r.targets, r.inference, marker='o', linewidth=4, markersize=8, label=to_label(r.name), color=COLORS[r.name])
            ax2.plot(r.targets, r.attack, marker='o', linewidth=4, markersize=8, label=to_label(r.name), color=COLORS[r.name])
            ax3.plot(r.targets, r.size, marker='o', linewidth=4, markersize=8, label=to_label(r.name), color=COLORS[r.name])

            print('{0} & {1:.4f} & {2:.4f}'.format(r.name, geometric_mean(r.inference), geometric_mean(r.attack)))

        for r in sim_results:
            for s in sim_results:
                test_result = stats.ttest_ind(r.attack, s.attack, equal_var=False)
                print('{0} vs {1}: {2:.4f}'.format(r.name, s.name, test_result.pvalue))

        ax1.set_xlabel('Fraction of Measurements')
        ax2.set_xlabel('Fraction of Measurements')
        ax3.set_xlabel('Fraction of Measurements')
        
        ax1.set_ylabel('Inference Accuracy')
        ax2.set_ylabel('Attack Accuracy')
        ax3.set_ylabel('Message Size')

        ax1.set_title('Inference Accuracy for Transmit Policies')
        ax2.set_title('Attacker Accuracy for Transmit Policies')
        ax3.set_title('Average Message Sizes')

        ax1.legend()
        ax2.legend()

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
        
        policy_name = serialized['policy']['name']
        if policy_name == 'adaptive':
            policy_name = '{0} {1}'.format(policy_name, serialized['policy']['compression_name'])

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
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    sim_results = list(map(extract_results, args.policy_folders))
    plot(sim_results, output_file=args.output_file)

