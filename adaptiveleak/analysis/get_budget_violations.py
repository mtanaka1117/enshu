from argparse import ArgumentParser
from typing import Dict, Tuple

from adaptiveleak.analysis.plot_utils import iterate_policy_folders, dataset_label
from adaptiveleak.utils.constants import POLICIES, ENCODING
from adaptiveleak.utils.file_utils import read_json_gz, iterate_dir


def extract_energy(folder: str) -> Tuple[str, Dict[int, int]]:
    """
    Creates a map from collection percentage to energy consumpion.
    """
    result: Dict[int, float] = dict()

    name = ''

    for path in iterate_dir(folder, '.*json.gz'):
        serialized = read_json_gz(path)

        num_collected = len(serialized['num_bytes'])
        total_count = serialized['count']

        rate = round(serialized['policy']['collection_rate'], 2)
        name = '{0}_{1}'.format(serialized['policy']['policy_name'].lower(), serialized['policy']['encoding_mode'].lower())

        result[int(rate * 100)] = int(num_collected < total_count)

    return name, result


def make_table(sim_results: Dict[str, Dict[int, int]]):
    for policy_name, policy_results in sim_results.items():
        total_count = len(policy_results)
        violated_count = sum(policy_results.values())
        print('{0} & {1}/{2} ({3:.4f})'.format(policy_name, violated_count, total_count, violated_count / total_count))

        if violated_count > 0:
            print('{0} : {1}'.format(policy_name, policy_results))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dates', type=str, nargs='+', required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    policy_folders = list(iterate_policy_folders(args.dates, dataset=args.dataset))
    sim_results = {name: res for name, res in map(extract_energy, policy_folders) if len(name) > 0}

    make_table(sim_results)
