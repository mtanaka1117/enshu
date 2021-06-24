import numpy as np
import os.path
from argparse import ArgumentParser

from adaptiveleak.utils.file_utils import save_json
from adaptiveleak.utils.loading import load_data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    inputs, _ = load_data(dataset_name=args.dataset, fold='train')
    
    inputs = inputs.reshape(-1, inputs.shape[-1])  # [N * T, D]
    mean = np.average(inputs, axis=0).astype(float).tolist()  # [D]
    std = np.std(inputs, axis=0).astype(float).tolist()  # [D]

    output_path = os.path.join('..', 'datasets', args.dataset, 'distribution.json')
    dist = dict(mean=mean, std=std)
    
    save_json(dist, output_path)
