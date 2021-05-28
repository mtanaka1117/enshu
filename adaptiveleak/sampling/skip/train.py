import h5py
import os.path

from argparse import ArgumentParser

from skip_rnn import SkipRNN
from test import test_model


UPDATE_WEIGHTS = [2.25, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.1, 0.0]
TARGETS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

SAVE_FOLDER = 'saved_models'
RNN_UNITS = 30
NAME = 'skip-rnn'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--should-print', action='store_true')
    args = parser.parse_args()

    train_path = os.path.join('..', '..', 'datasets', args.dataset, 'train', 'data.h5')
    with h5py.File(train_path, 'r') as fin:
        train_inputs = fin['inputs'][:]

    val_path = os.path.join('..', '..', 'datasets', args.dataset, 'validation', 'data.h5')
    with h5py.File(val_path, 'r') as fin:
        val_inputs = fin['inputs'][:]

    for target, weight in zip(TARGETS, UPDATE_WEIGHTS):

        print('===== Starting {0:.2f} ====='.format(target))

        hypers = dict(rnn_units=RNN_UNITS, update_weight=weight, target=target, warmup=3)

        model = SkipRNN(hypers=hypers, name=NAME)

        save_folder, model_name = model.train(train_inputs=train_inputs,
                                              val_inputs=val_inputs,
                                              dataset_name=args.dataset,
                                              save_folder=SAVE_FOLDER,
                                              should_print=args.should_print)

        if args.should_print:
            print('Finished Training. Starting testing...')

        test_model(model_name=model_name,
                   dataset_name=args.dataset,
                   save_folder=save_folder)
