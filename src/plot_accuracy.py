import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
from collections import namedtuple
from typing import Any, Dict, List

from utils.file_utils import read_pickle_gz


SimResult = namedtuple('SimResult', ['inference', 'attack', 'targets', 'name'])

COLORS = {
    'random': '#ca0020',
    'all': '#f4a582',
    'adaptive fixed': '#92c5de',
    'adaptive pid': '#0571b0'
}


def to_label(label: str) -> str:
    return ' '.join(t.capitalize() for t in label.split())


def plot(sim_results: List[SimResult]):

    with plt.style.context('seaborn-ticks'):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

        for r in sim_results:
            ax1.plot(r.targets, r.inference, marker='o', linewidth=4, markersize=8, label=to_label(r.name), color=COLORS[r.name])
            ax2.plot(r.targets, r.attack, marker='o', linewidth=4, markersize=8, label=to_label(r.name), color=COLORS[r.name])

        ax1.set_xlabel('Fraction of Measurements')
        ax2.set_xlabel('Fraction of Measurements')
        
        ax1.set_ylabel('Inference Accuracy')
        ax2.set_ylabel('Attack Accuracy')

        ax1.set_title('Inference Accuracy for Transmit Policies')
        ax2.set_title('Attacker Accuracy for Transmit Policies')

        ax1.legend()
        ax2.legend()

        plt.show()


def extract_results(folder: str) -> SimResult:

    accuracy: Dict[float, Tuple[float, float]] = dict()

    for file_name in sorted(os.listdir(folder)):
        path = os.path.join(folder, file_name)
        serialized = read_pickle_gz(path)

        target = serialized['policy']['target']
        inference = serialized['accuracy']
        attack = serialized['attack']['test_accuracy']
        
        policy_name = serialized['policy']['name']
        if policy_name == 'adaptive':
            policy_name = '{0} {1}'.format(policy_name, serialized['policy']['compression_name'])

        accuracy[target] = (inference, attack)

    inference_list: List[float] = []
    attack_list: List[float] = []
    target_list: List[float] = []

    for target, (inf, att) in sorted(accuracy.items()):
        target_list.append(target)
        inference_list.append(inf)
        attack_list.append(att)

    return SimResult(inference=inference_list,
                     attack=attack_list,
                     targets=target_list,
                     name=policy_name)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--policy-folders', type=str, nargs='+', required=True)
    args = parser.parse_args()

    sim_results = list(map(extract_results, args.policy_folders))
    plot(sim_results)

