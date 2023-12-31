import h5py
import os.path
import numpy as np

from argparse import ArgumentParser

from adaptiveleak.utils.file_utils import read_json, make_dir
from adaptiveleak.utils.data_utils import array_to_fp
from adaptiveleak.utils.loading import load_data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset.')
    parser.add_argument('--num-seq', type=int, required=True, help='The number of sequences to write.')
    parser.add_argument('--offset', type=int, default=0, required=True, help='The sequence offset.')
    parser.add_argument('--is-msp', action='store_true', help='Whether to serialize for the MSP430.')
    args = parser.parse_args()

    base = os.path.join('datasets', args.dataset)
    inputs, outputs = load_data(dataset_name=args.dataset, fold='test')

    # Get a random sample of inputs
    rand = np.random.RandomState(seed=5494)
    sample_idx = np.arange(inputs.shape[0])
    rand.shuffle(sample_idx)

    indices = sample_idx[args.offset:args.offset + args.num_seq]

    inputs = inputs[indices]
    outputs = outputs[indices]

    # Make the output folder
    make_dir(os.path.join(base, 'mcu'))

    # Write the outputs to h5 files
    with h5py.File(os.path.join(base, 'mcu', 'data.h5'), 'w') as fout:
        input_ds = fout.create_dataset(name='inputs', shape=inputs.shape, dtype='f')
        input_ds.write_direct(inputs)

        output_ds = fout.create_dataset(name='output', shape=outputs.shape, dtype='i')
        output_ds.write_direct(outputs)

    # Load the quantization parameters
    quantization_path = os.path.join(base, 'quantize.json')
    quantization = read_json(quantization_path)

    width = quantization['width']
    precision = quantization['precision']

    # Quantize the data
    fixed_point = array_to_fp(inputs, width=width, precision=precision)

    # Flatten the data
    fixed_point = fixed_point.reshape(-1)

    # Convert data into a 1d array in string format
    data_string = '{{ {0} }}'.format(','.join(map(str, fixed_point)))

    # Write to a C header file
    with open('data.h', 'w') as fout:
        fout.write('#include <stdint.h>\n')

        if args.is_msp:
            fout.write('#include <msp430.h>\n')

        fout.write('#include "utils/fixed_point.h"\n')

        fout.write('#ifndef DATA_H_\n')
        fout.write('#define DATA_H_\n')

        num_values = len(fixed_point)

        fout.write('#define MAX_NUM_SEQ {0}u\n'.format(args.num_seq))
        fout.write('static const uint32_t DATASET_LENGTH = {0}ul;\n'.format(num_values))

        if args.is_msp:
            fout.write('#pragma PERSISTENT(DATASET)\n')

        fout.write('static FixedPoint DATASET[{0}] = {1};\n'.format(len(fixed_point), data_string))

        fout.write('#endif\n')
