import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import Counter, defaultdict, OrderedDict, namedtuple
from typing import Dict, List, DefaultDict, Optional, Tuple

from adaptiveleak.analysis.plot_utils import COLORS, PLOT_STYLE, LINE_WIDTH, MARKER, MARKER_SIZE, to_label
from adaptiveleak.analysis.plot_utils import LEGEND_FONT, AXIS_FONT, PLOT_SIZE, TITLE_FONT
from adaptiveleak.analysis.plot_utils import iterate_policy_folders, dataset_label
from adaptiveleak.utils.constants import POLICIES, SMALL_NUMBER
from adaptiveleak.utils.file_utils import read_json_gz, iterate_dir



def plot(information_results: DefaultDict[str, Dict[float, float]], dataset: str, output_file: Optional[str]):
    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots(figsize=PLOT_SIZE)

        names: List[str] = []
        policy_values: List[float] = []

        for name in POLICIES:
            encodings = ['standard', 'padded', 'group'] if name not in ('uniform', 'random') else ['standard']

            for encoding in encodings:

                policy_name = '{0}_{1}'.format(name, encoding)

                if (policy_name not in information_results) and (name not in information_results):
                    continue

                if name in information_results:
                    policy_name = name

                information = information_results[policy_name]

                energy = sorted(information.keys())
                values = [information[e] for e in energy]

                ax.plot(energy, values, label=to_label(policy_name), color=COLORS[policy_name], linewidth=LINE_WIDTH, marker=MARKER, markersize=MARKER_SIZE)

                names.append(policy_name)
                policy_values.append((np.median(values), np.max(values)))

        ax.legend(fontsize=LEGEND_FONT, loc='center')

        ax.set_title('Empirical Mutual Information between Message Size and Label on the {0} Dataset'.format(dataset_label(dataset)), fontsize=TITLE_FONT)
        ax.set_xlabel('Energy Budget (mJ)', fontsize=AXIS_FONT)
        ax.set_ylabel('Empirical Mutual Information (nits)', fontsize=AXIS_FONT)

        print(' & '.join(names))
        print(' & '.join(map(lambda t: '{0:.2f} ({1:.2f})'.format(t[0], t[1]), policy_values)))

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dates', type=str, required=True, nargs='+')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    information_results: DefaultDict[str, Dict[float, float]] = defaultdict(dict)

    for folder in iterate_policy_folders(args.dates, dataset=args.dataset):
        for sim_file in iterate_dir(folder, pattern='.*json.gz'):
            model = read_json_gz(sim_file)

            if model['policy']['encoding_mode'].lower() in ('single_group', 'group_unshifted', 'pruned'):
                continue

            name = '{0}_{1}'.format(model['policy']['policy_name'].lower(), model['policy']['encoding_mode'].lower())
            energy_per_seq = model['policy']['energy_per_seq']

            mutual_information = model['mutual_information']['norm_mutual_information']

            information_results[name][energy_per_seq] = mutual_information

    plot(information_results, dataset=args.dataset, output_file=args.output_file)
