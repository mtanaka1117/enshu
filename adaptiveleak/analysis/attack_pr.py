import numpy as np
from argparse import ArgumentParser
from adaptiveleak.utils.file_utils import read_json_gz


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--label', type=int, required=True)
    args = parser.parse_args()

    log = read_json_gz(args.file)
    test = log['attack']['test']
    idx = args.label

    precision = []
    recall = []

    for confusion_mat in test['confusion_mat']:
        true_pos = confusion_mat[idx][idx]
        total_pos = np.sum(confusion_mat[idx])
        total_pred = np.sum([array[idx] for array in confusion_mat])

        precision.append(true_pos / total_pos)
        recall.append(true_pos / total_pred)

    print(precision)
    print(recall)

    print('Precision: {0}'.format(np.average(precision)))
    print('Recall: {0}'.format(np.average(recall)))
