import h5py
import os.path
import numpy as np

from argparse import ArgumentParser

from adaptiveleak.utils.file_utils import save_json_gz
from neural_network import MODEL_FILE_FMT, TEST_LOG_FMT
from skip_rnn import SkipRNN


def test_model(model_name: str, save_folder: str, dataset_name: str):
    test_path = os.path.join('..', '..', 'datasets', dataset_name, 'test', 'data.h5')
    with h5py.File(test_path, 'r') as fin:
        test_inputs = fin['inputs'][:]
        test_labels = fin['output'][:]

    if len(test_inputs.shape) == 2:
        test_inputs = np.expand_dims(test_inputs, axis=-1)

    model_path = os.path.join(save_folder, MODEL_FILE_FMT.format(model_name))
    model = SkipRNN.restore(model_file=model_path)

    test_result = model.test(test_inputs=test_inputs, batch_size=None)

    test_log_path = os.path.join(save_folder, TEST_LOG_FMT.format(model_name))
    save_json_gz(test_result, test_log_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model-file', type=str, required=True)
    args = parser.parse_args()

    save_folder, model_name = os.path.split(args.model_file)
    model_name = model_name.split('.')[0]

    test_model(model_name=model_name,
               save_folder=save_folder,
               dataset_name=args.dataset)
