import numpy as np
import os.path
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import namedtuple
from typing import List, Optional

from adaptiveleak.analysis.plot_utils import PLOT_STYLE, LINE_WIDTH, MARKER_SIZE, AXIS_FONT, TITLE_FONT
from adaptiveleak.utils.file_utils import read_json_gz, iterate_dir, read_json


EnergyResult = namedtuple('EnergyResults', ['num_features', 'energy', 'energy_std'])


def plot(group_results: EnergyResult, standard_results: EnergyResult, output_file: Optional[str]):

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots()

        ax.errorbar(group_results.num_features, group_results.energy, linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE, label='AGE', color='#08519c')
        ax.errorbar(standard_results.num_features, standard_results.energy, linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE, label='Standard', color='#9ecae1')

        ax.legend(fontsize=AXIS_FONT)
        ax.set_xlabel('Number of Values', fontsize=AXIS_FONT)
        ax.set_xticklabels(list(map(int, ax.get_xticks())), fontsize=AXIS_FONT)

        ax.set_ylabel('Energy (mJ)', fontsize=AXIS_FONT)
        ax.set_yticklabels([round(y, 2) for y in ax.get_yticks()], fontsize=AXIS_FONT)

        ax.set_title('Energy to Encode Measurements', fontsize=TITLE_FONT)

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file, transparent=True, bbox_inches='tight')


def get_energy(folder: str) -> EnergyResult:
    feature_count_list: List[int] = []
    energy_list: List[float] = []
    std_list: List[float] = []

    for path in iterate_dir(folder):
        try:
            num_features = int(os.path.split(path)[-1])
        except ValueError:
            continue

        energy_path = os.path.join(path, 'energy.json')
        energy_results = read_json(energy_path)
        avg_energy = np.average(energy_results['energy'])
        std_energy = np.std(energy_results['energy'])

        energy_list.append(avg_energy)
        std_list.append(std_energy)
        feature_count_list.append(num_features)

    # Sort by the number of features
    feature_count_list, energy_list = zip(*sorted(zip(feature_count_list, energy_list)))

    return EnergyResult(num_features=feature_count_list,
                        energy=energy_list,
                        energy_std=std_list)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    group_folder = os.path.join(args.folder, 'group')
    standard_folder = os.path.join(args.folder, 'standard')

    group_energy = get_energy(group_folder)
    standard_energy = get_energy(standard_folder)

    plot(group_results=group_energy,
         standard_results=standard_energy,
         output_file=args.output_file)
