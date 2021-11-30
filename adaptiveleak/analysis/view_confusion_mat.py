from argparse import ArgumentParser

from adaptiveleak.utils.file_utils import read_json_gz


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--log-path', type=str, required=True, help='Path to the target jsonl.gz log.')
    args = parser.parse_args()

    attack_log = read_json_gz(args.log_path)['attack']['test']
    print('Confusion Matrices for all 5 folds.')
    print(attack_log['confusion_mat'])
