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


THRESHOLD = 0.01


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='The name of the experiment log directory.')
    parser.add_argument('--datasets', type=str, required=True, nargs='+', help='The names of all datasets to analyze.')
    args = parser.parse_args()

    print('Num Datasets: {0}'.format(len(args.datasets)))
    print('==========')

    test_results: DefaultDict[str, int] = defaultdict(int)
    budget_counts: DefaultDict[str, int] = defaultdict(int)

    for dataset in args.datasets:
        for folder in iterate_policy_folders([args.folder], dataset=dataset):
            for sim_file in iterate_dir(folder, pattern='.*json.gz'):
                model = read_json_gz(sim_file)

                if model['policy']['encoding_mode'].lower() in ('single_group', 'group_unshifted', 'padded', 'pruned'):
                    continue

                name = '{0}_{1}'.format(model['policy']['policy_name'].lower(), model['policy']['encoding_mode'].lower())
                energy_per_seq = model['policy']['energy_per_seq']

                p_value = model['mutual_information']['p_value']
                num_trials = model['mutual_information']['num_trials']

                upper_bound = p_value + 1.96 * (1.0 / (2 * np.sqrt(num_trials)))

                test_results[name] += int(upper_bound < THRESHOLD)
                budget_counts[name] += 1

    policy_names: List[str] = []
    policy_values: List[Tuple[int, int]] = []

    for name in POLICIES:
        encodings = ['standard', 'group'] if name not in ('uniform', 'random') else ['standard']

        for encoding in encodings:

            policy_name = '{0}_{1}'.format(name, encoding)

            if (policy_name not in test_results):
                continue

            num_nontrivial = test_results[policy_name]
            count = budget_counts[policy_name]

            policy_names.append(policy_name)
            policy_values.append((num_nontrivial, count))

    print(' & '.join(policy_names))
    print(' & '.join(map(lambda t: '{0} / {1}'.format(t[0], t[1]), policy_values)))
