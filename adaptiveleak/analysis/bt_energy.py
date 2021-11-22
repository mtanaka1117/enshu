import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List, Optional

from adaptiveleak.analysis.plot_utils import PLOT_STYLE, LINE_WIDTH, PLOT_SIZE, AXIS_FONT, TITLE_FONT, MARKER_SIZE, ANNOTATE_FONT
from adaptiveleak.utils.file_utils import read_json, iterate_dir


def plot_energy(energy: List[float], num_bytes: List[int], output_file: Optional[str]):
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=PLOT_SIZE)

        zipped = sorted(zip(num_bytes, energy))
        num_bytes, energy = zip(*zipped)

        ax.plot(num_bytes, energy, linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE, color='k')

        for i, (count, value) in enumerate(zip(num_bytes, energy)):
            xmargin = -32 if i % 2 == 0 else -10
            ymargin = 1 if i % 2 == 0 else -1.2

            if (i == len(num_bytes) - 1):
                xmargin = -25
                ymargin = -1.5

            ax.annotate('{0:.2f}'.format(value),
                        xy=(count, value),
                        xytext=(count + xmargin, value + ymargin),
                        fontsize=ANNOTATE_FONT)

        ax.set_xticklabels(list(map(int, ax.get_xticks())), fontsize=AXIS_FONT)
        ax.set_yticklabels(ax.get_yticks(), fontsize=AXIS_FONT)

        ax.set_xlabel('Number of Sent Bytes', fontsize=AXIS_FONT)
        ax.set_ylabel('Energy (mJ)', fontsize=AXIS_FONT)
        ax.set_title('Median Energy for BLE Connection and Transmission', fontsize=TITLE_FONT)

        if output_file is not None:
            plt.savefig(output_file, transparent=True, bbox_inches='tight')
        else:
            plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    num_bytes: List[int] = []
    energy: List[float] = []

    for bt_folder in iterate_dir(args.input_folder):
        try:
            byte_count = int(bt_folder.split(os.sep)[-1])
        except ValueError:
            continue

        energy_path = os.path.join(bt_folder, 'energy.json')
        energy_results = read_json(energy_path)

        energy.append(energy_results['median'])
        num_bytes.append(byte_count)

    plot_energy(energy=energy, num_bytes=num_bytes, output_file=args.output_file)
