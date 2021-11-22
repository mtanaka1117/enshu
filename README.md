# Adaptive Group Encoding
This repository contains the implementation of Adaptive Group Encoding (AGE), a system for protecting adaptive sampling algorithms from leaking information through communication patterns on low-power devices. This work was accepted into ASPLOS 2022. The repository has the following general structure. Note that most of code supports the simulator framework, and the paths below all lie within the `adaptiveleak` directory.

1. `analysis`: Scripts to analysis experiment results.
2. `attack`: Script to train the attack classifier model.
3. `device`: Holds server code for experiments with the TI MSP430.
4. `energy_systems`: Manages energy traces to track energy consumption in the simulator framework.
5. `msp`: Contains the TI MSP430FR5994 implementation of all encoding and sampling strategies.
6. `plots`: Holds all plots for the included experimental results.
7. `skip_rnn`: Implements a Skip RNN sampling policy.
8. `traces`: Contains the pre-collected energy traces from a TI MSP430 FR5994.
9. `unit_tests`: A suite of unit tests for various aspects of the system.
10. `utils`: Holds a set of utility functions for actions such as encryption and encoding.
11. `policies.py`: Implements all sampling policies and encoding strategies.
12. `sensor.py`: Represents the simulated sensor.
13. `server.py`: Contains the simulated server.
14. `serialize_policy`: Converts a policy to a C header file for deployment onto a microcontroller (MCU).
15. `simulator.py`: The simulator entry point.

## Simulator

The simulator framework executes sub-sampling policies standard machines by representing sensors and servers as independent processes. This framework is written entirely in Python 3 and runs on pre-collected datasets.

### Installation
You can install the Python package (and associated dependencies) using `pip`. To avoid version conflicts, it is best to install the package inside a virtual environment. You may create an environment called `adaptiveleak-env` using the command below.
```
python3 -m venv adaptiveleak-env
```
You can then enter the environment using the following command.
```
. adaptiveleak-env/bin/activate
```
To exit the environment, run the command `deactivate`. **The entirety of the simulator must run inside the virtual environment**. Once inside the environment, install the package using the commands below. Note this must run in the root directory of the repository.
```
pip3 install --upgrade pip
pip3 install -e .
```

### Downloading Datasets

### Running Sampling Policies

### Attack Classifier
After policy execution, you can execute a more practical attack using a statistical classifier. This model is an AdaBoost ensemble of decision trees that uses the message sizes from consecutive sequences to predict the corresponding event label. To train a classifier for a set of simulator results, navigate into the `adaptiveleak/attack` directory and run the following command.
```
python train.py --policy <policy-name> --encoding <encoding-name> --dataset <dataset-name> --folder <experiment-name> --window-size <window-size> --num-samples <num-samples>
```
For a longer description of each option, run `python train.py --help`. The `folder` option should be the name of the folder containing the experimental logs in `saved_models/<dataset-name>`. These logs are the results of the previous section. Note that the folder name defaults to the current date.

The training process uses 5-fold stratified cross evaluation. The results get automatically stored in the evaluation logs for each sampling policy, encoding algorithm, and collection rate. Each entry in the serialized result is a list of 5 elements following the 5-fold evaluation.

The script `analysis/plot_attack.py` analyzes the attack classification results. You can execute this script using the command below.
```
python plot_attack.py --folder <experiment-name> --dataset <dataset-name> --output-file [<optional-output-path>]
```
Executing `python plot_attack.py --help` provides longer descriptions of each argument. The provided `--folder` follows the same convention as the `train.py` script above. To analyze the previous results, use the same `folder`. The `plot_attack.py` script will show the median attack accuracy across all 5 evaluation folds for each energy budget. The script also prints out the average and maximum attack accuracy across all constraints. For the Skip RNN, these values correspond to the Attack results in Table 5 of the paper.

The script `analysis/plot_all_attacks.py` will show the median and maximum attack accuracy values for multiple datasets. You may run this script using the command below.
```
python plot_all_attacks.py --folder <experiment-name> --datasets <list-of-dataset-names> --output-file [<optional-output-path>]
```
The result is a bar chart that shows the median, IQR, and maximum attack accuracy values for each provided dataset. Running this command with all 9 provided datasets yields Figure 6 in the paper.

### Unit Tests

## Hardware (TI MSP430)

### Setup

### Running Experiments

### Analysis
