import matplotlib.pyplot as plt
import os
import numpy as np
from argparse import ArgumentParser
from functools import partial
from scipy import stats
from collections import namedtuple, OrderedDict
from typing import Any, Dict, List, Optional

from adaptiveleak.utils.constants import POLICIES, ENCODING
from adaptiveleak.utils.file_utils import read_json_gz
from adaptiveleak.analysis.plot_utils import COLORS, to_label, geometric_mean, MARKER, MARKER_SIZE, LINE_WIDTH, PLOT_STYLE
from adaptiveleak.analysis.plot_utils import PLOT_SIZE, AXIS_FONT, LEGEND_FONT, TITLE_FONT
from adaptiveleak.analysis.plot_utils import extract_results, iterate_policy_folders



def plot(sim_results: Dict[str, Dict[float, float]], dataset_name: str, output_file: Optional[str]):

    with plt.style.context(PLOT_STYLE):
        fig, (ax1, ax2) = plt.subplots(figsize=(2 * PLOT_SIZE[0], PLOT_SIZE[1]), nrows=1, ncols=2)

        for name in POLICIES:
            for encoding in ['standard', 'group']:
                policy_name = '{0}_{1}'.format(name, encoding)

                if policy_name not in sim_results:
                    continue

                model_results = sim_results[policy_name]

                energy_budgets = list(sorted(model_results.keys()))

                # Get the accuracy and F1 scores for each target
                accuracy: List[float] = []
                f1: List[float] = []

                all_accuracy: List[float] = []
                all_f1: List[float] = []

                for budget in energy_budgets:
                    if 'test' not in model_results[budget]:
                        accuracy.append(0.0)
                        f1.append(0.0)
                    else:
                        budget_accuracy = model_results[budget]['test']['accuracy']
                        accuracy.append(geometric_mean(budget_accuracy))
                        all_accuracy.extend(budget_accuracy)

                        budget_f1 = model_results[budget]['test']['f1']
                        f1.append(geometric_mean(budget_f1))
                        all_f1.extend(budget_f1)

                ax1.plot(energy_budgets, accuracy, marker=MARKER, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=to_label(policy_name), color=COLORS[policy_name])
                ax2.plot(energy_budgets, f1, marker=MARKER, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=to_label(policy_name), color=COLORS[policy_name])

                if len(all_accuracy) > 0:
                    print('{0} & {1:.2f}\\% & {2:.2f} \\\\'.format(to_label(policy_name), geometric_mean(all_accuracy), geometric_mean(all_f1)))

        ax1.set_xlabel('Energy Budget (mJ)', fontsize=AXIS_FONT)
        ax1.set_ylabel('Mean Accuracy', fontsize=AXIS_FONT)
        ax1.set_title('Attacker Accuracy on the {0} Dataset'.format(dataset_name.capitalize()), fontsize=TITLE_FONT)

        ax2.set_xlabel('Energy Budget (mJ)', fontsize=AXIS_FONT)
        ax2.set_ylabel('Mean Macro F1 Score', fontsize=AXIS_FONT)
        ax2.set_title('Attacker F1 Score on the {0} Dataset'.format(dataset_name.capitalize()), fontsize=TITLE_FONT)

        ax1.legend(fontsize=LEGEND_FONT)

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file, bbox_inches='tight')
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dates', type=str, nargs='+', required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    extract_fn = partial(extract_results, field='attack', aggregate_mode=None, default_value=dict(test_accuracy=0.0))
    policy_folders = iterate_policy_folders(args.dates, dataset=args.dataset)

    sim_results = {name: res for name, res in map(extract_fn, policy_folders)}
    plot(sim_results, output_file=args.output_file, dataset_name=args.dataset)

