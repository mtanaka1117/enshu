# Skip RNN
This folder contains code to train and test [Skip RNNs (Campos et al. 2017)](https://arxiv.org/abs/1708.06834) for adaptive sampling. Training Skip RNNs from scratch can take a long time. We include trained Skip RNNs in the `saved_models` for each dataset.

## Installation
This module depends on Tensorflow 2. The simulator implements Skip RNNs using Numpy, and we do not list this package as a dependency due to its sparse use and large size. You can install Tensorflow using `pip`. Run the `pip install tensorflow` command within the virtual environment.

## Training
The file `train.py` launches model training. You may execute the training procedure using the following command.
```
python train.py --dataset <dataset-name> --should-print
```
This command will train `8` Skip RNNs, one for each target fraction in `range(0.3, 1.01, 0.1)`. To modify this behavior, change the array `TARGETS` in the file `train.py`. The training results are stored in the directory `saved_models/<dataset-name>`. The pickle files contain the model parameters, and the collection rate is listed in the file name.

The Skip RNN implementation is in the file `skip_rnn.py`.

## Testing
The file `test.py` executes model testing using the command below.
```
python test.py --dataset <dataset-name> --model-file <path-to-pickle>
```
The `model file` should be the path to the saved model parameters (`.pkl.gz` file) from the training step.

## Integrate with Simulator
If you want to integrate the newly trained Skip RNN into the simulator, copy the saved pickle file into the folder `adaptiveleak/saved_models/<dataset-name>/skip_rnn`. The simulator sources model parameters from this folder. **Note that this process will overwrite the model currently in the directory.**
