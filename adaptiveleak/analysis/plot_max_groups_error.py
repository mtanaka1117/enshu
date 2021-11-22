import os.path
import matplotlib.pyplot as plt
from typing import List

from adaptiveleak.analysis.plot_utils import PLOT_STYLE, AXIS_FONT, TITLE_FONT, LEGEND_FONT, MARKER_SIZE, LINE_WIDTH, PLOT_SIZE
from adaptiveleak.utils.file_utils import iterate_dir, read_json_gz


def get_avg_error(folder: str) -> float:
    error_list: List[float] = []

    for path in iterate_dir(folder, pattern=r'.*json.gz'):
        error = read_json_gz(path)['mae']
        error_list.append(error)

    return sum(error_list) / len(error_list)


def get_percent_change(errors: List[float]) -> float:
    min_val = min(errors)
    max_val = max(errors)

    return (max_val - min_val) / ((max_val + min_val) / 2) * 100


if __name__ == '__main__':
    char_folder = '../saved_models/trajectories'
    tiselac_folder = '../saved_models/tiselac'
    output_file = '../plots/2021-07-26/sensitivity.pdf'

    # Get the 4, 6, and 8 group errors
    folders = ['groups_4', 'tiny', 'groups_8']

    char_heuristic: List[float] = []
    char_deviation: List[float] = []
    tiselac_heuristic: List[float] = []
    tiselac_deviation: List[float] = []

    for folder in folders:
        char_heuristic.append(get_avg_error(os.path.join(char_folder, folder, 'adaptive_heuristic_group')))
        char_deviation.append(get_avg_error(os.path.join(char_folder, folder, 'adaptive_deviation_group')))
        tiselac_heuristic.append(get_avg_error(os.path.join(tiselac_folder, folder, 'adaptive_heuristic_group')))
        tiselac_deviation.append(get_avg_error(os.path.join(tiselac_folder, folder, 'adaptive_deviation_group')))

    # Calculate Percent Changes
    print('Characters Heuristic: {0:.5f}'.format(get_percent_change(char_heuristic)))
    print('Characters Deviation: {0:.5f}'.format(get_percent_change(char_deviation)))
    print('Tiselac Heuristic: {0:.5f}'.format(get_percent_change(tiselac_heuristic)))
    print('Tiselac Deviation: {0:.5f}'.format(get_percent_change(tiselac_deviation)))

    xs: List[int] = [4, 6, 8]

    with plt.style.context(PLOT_STYLE):
        fig, ax1 = plt.subplots(figsize=PLOT_SIZE)
        ax2 = ax1.twinx()
    
        ax1.plot(xs, char_heuristic, marker='o', color='#0571b0', label='Chars. Heuristic AGE', markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
        ax1.plot(xs, char_deviation, marker='o', color='#92c5de', label='Chars. Deviation AGE', markersize=MARKER_SIZE, linewidth=LINE_WIDTH)

        ax2.plot(xs, tiselac_heuristic, marker='o', color='#ca0020', label='Tiselac Heuristic AGE', markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
        ax2.plot(xs, tiselac_deviation, marker='o', color='#f4a582', label='Tiselac Deviation AGE', markersize=MARKER_SIZE, linewidth=LINE_WIDTH)

        ax1.legend(loc='center left', fontsize=LEGEND_FONT)
        ax2.legend(loc='center right', fontsize=LEGEND_FONT)

        ax1.set_ylabel('Characters MAE', fontsize=AXIS_FONT, color='#0571b0')
        ax1.set_yticklabels([round(y, 6) for y in ax1.get_yticks()], fontsize=AXIS_FONT, color='#0571b0')

        ax2.set_ylabel('Tiselac MAE', fontsize=AXIS_FONT, color='#ca0020')
        ax2.set_yticklabels([round(y, 6) for y in ax2.get_yticks()], fontsize=AXIS_FONT, color='#ca0020')

        ax1.set_xlabel('Maximum # Groups ($G_0$)', fontsize=AXIS_FONT)
        ax1.set_xticks(xs)
        ax1.set_xticklabels(list(map(int, ax1.get_xticks())), fontsize=AXIS_FONT)

        ax1.set_title('Average MAE for Varying Maximum Group Limits', fontsize=TITLE_FONT)

        plt.savefig(output_file, bbox_inches='tight', transparent=True)
