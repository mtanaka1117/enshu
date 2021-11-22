import matplotlib.pyplot as plt
import scipy.stats as stats
import os.path
from argparse import ArgumentParser
from collections import defaultdict
from typing import DefaultDict, Dict, List

from adaptiveleak.analysis.plot_utils import PLOT_STYLE, PLOT_SIZE, TITLE_FONT, AXIS_FONT, to_label
from adaptiveleak.utils.file_utils import read_json_gz


CLASS_LABELS = {
    'epilepsy': ['Epilepsy', 'Walking', 'Running', 'Sawing']
}



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--log-file', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    sim_log = read_json_gz(args.log_file)
    policy_name = sim_log['policy']['policy_name']

    # Get the labels and number of bytes
    labels: List[int] = sim_log['labels']
    num_bytes: List[int] = sim_log['num_bytes']

    # Group the byte distributions by label
    label_counts: DefaultDict[int, List[int]] = defaultdict(list)

    for label, byte_count in zip(labels, num_bytes):
        label_counts[label].append(byte_count)

    # Get the number of classes
    num_classes = max(labels) + 1

    # Execute the t-tests
    test_results: List[List[int]] = []

    for label1 in range(num_classes):
        label1_results: List[int] = []

        for label2 in range(num_classes):
            test_stat, p_value = stats.ttest_ind(label_counts[label1], label_counts[label2], equal_var=False)
            label1_results.append(p_value)

        test_results.append(label1_results)

    # Read the data-set name to retrieve the labels
    path_tokens = args.log_file.split(os.sep)
    dataset_name = path_tokens[-4].lower()
    axis_labels = CLASS_LABELS[dataset_name]

    # Plot the result
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=PLOT_SIZE)

        cax = ax.matshow(test_results, cmap=plt.cm.magma)
        fig.colorbar(cax)

        # Annotate the plot
        for row in range(num_classes):
            for col in range(num_classes):
                p_value = test_results[row][col]
                text = '{0:.3f}'.format(p_value) if p_value >= 1e-3 else '<0.001'

                ax.text(row, col, text, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

        ax.xaxis.set_ticks_position('bottom')

        ticks = list(range(num_classes))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.set_xticklabels(axis_labels, fontsize=AXIS_FONT)
        ax.set_yticklabels(axis_labels, fontsize=AXIS_FONT)

        ax.set_title('Welch\'s T-Test P-Values for the {0} Policy\'s Message Size Distributions'.format(to_label(policy_name)), fontsize=TITLE_FONT)

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
