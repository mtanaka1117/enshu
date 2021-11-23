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
10. `utils`: Holds a set of utility functions for actions such as encryption and encoding. The README in this folder contains more information on the implemented functionality.
11. `fit_threshold.py`: Script to train threshold-based adaptive sampling policies for various energy budgets.
12. `policies.py`: Implements all sampling policies and encoding strategies.
13. `sensor.py`: Represents the simulated sensor.
14. `server.py`: Contains the simulated server.
15. `serialize_dataset.py`: Converts a dataset into a C header file for deployment onto a microcontroller (MCU).
16. `serialize_policy`: Converts a policy to a C header file for deployment onto a MCU.
17. `simulator.py`: The simulator entry point.

## Simulator
The simulator framework executes sub-sampling policies standard machines by representing sensors and servers as independent processes. This framework is written entirely in Python 3 and runs on pre-collected datasets.

### Installation
This repository requires Python 3. You can install the Python package (and associated dependencies) using `pip`. To avoid version conflicts, it is best to install the package inside a virtual environment. You may create an environment called `adaptiveleak-env` using the command below.
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

### Naming
There are two naming conventions in the code repository that differ from the paper. First, in the code, the `AGE` encoding system is called `group`. Second, the `Linear` policy in the paper is called the adaptive `heuristic` policy in the code.

### Running Sampling Policies
The entry point for the simulator is the script `adaptiveleak/simulator.py`. You must navigate into the `adaptiveleak` directory to run the script. This code has many command line options which are best viewed when running `python simulator.py --help`. You can execute the simulator on a single dataset and policy using the command below. In the next paragraph, we describe a utility that runs all policies on a single dataset.
```
python simulator.py --dataset <dataset-name> --encoding <encoding-name> --encryption <encryption-type> --collection-rate <budget> --should-print
```
The collection rate is the target fraction of elements in each sequence to capture; the budget is set at the `Uniform` policy's energy consumption at this fraction. You can specify a range of elements by providing three values (space-separated) in the form `<min> <max> <step>`. The results in the paper use `--colelction-rate 0.3 1.0 0.1`. As a note, the encoding algorithm `group` is the full `AGE` system. The dataset name is the name of the folder in `datasets` (e.g. `datasets/<dataset-name>`) containing the data files. The shell script `adaptiveleak/run_simulator.sh` executes all policies on the dataset passed as a command line argument (shown below).
```
./run_simlator.sh <dataset-name>
```
This command can take a few minutes to run, especially for the larger datasets (`uci_har`, `mnist`, `tiselac`). The `epilepsy` dataset is relatively small and represents a good starting point.

After execution, the results are automatically stored in the folder `saved_models/<dataset-name>/<date>`. There will be a folder in this directory for the sampling policy and the encoding algorithm. Keep note of the date, as you will use this value to reference these results during the analysis phase (below).

### Analyzing Experimental Results
The `adaptiveleak/analysis` folder contains a few scripts to process the results of each experiment. This section describes how to compute the reconstruction error, as well as the mutual information between message size and event label.

#### Reconstruction Error
The script `adaptiveleak/analysis/plot_error.py` displays the arithmetic mean reconstruction error for each budget in the executed experiment. You can run this script by navigating to the directory `adaptiveleak/analysis` folder and running the command below.
```
python plot_error.py --folder <experiment-name> --dataset <dataset-name> --metric <metric-name>
```
The arguments are described when running `python plot_error.py --help`. The `--folder` should be the date of the experiment as produced by the execution step in the last section. The code will look for the folder `adaptiveleak/saved_models/<dataset>/<folder>` and retrieve the results from this directory.

The script will produce a plot showing the error for each constraint. The code will also print out the arithmetic mean error (across all constraints) for each policy. When the provided `metric` is `mae`, the printed error values should align with the results in Table 3 of the paper. Note that the plot does not include `padded` policies due to their high error.

#### Mutual Information
We measure the theoretical information leakage on each task using the mutual information between message sizes and event labels. You can compute these results using the script `adaptiveleak/analysis/leakage_test.py`. This script also executes permutation tests to measure the significance of the observed empirical relationship. The command below describes how to run the script. You must be in the `adaptiveleak/analysis` directory.
```
python leakage_test.py --folder <experiment-name> --dataset <dataset-name> --trials <num-perm-trials>
```
The `folder` and `dataset` arguments are the same with previous scripts. The `trials` argument controls the number of randomized permutation trials. Larger values create higher-confidence results at the cost of greater computation; using a large number of trials can take a long time. To just compute the mutual information, set the `trials` argument to `1`.

The script `adaptiveleak/analysis/mutual_information.py` displays the mutual information results for every budget. You can run this command as shown below.
```
python mutual_information.py --folder <experiment-name> --dataset <dataset-name>
```
The plot shows the mutual information for each constraint. The script also prints out the median and maximum mutual information across all constraints. The printed values should align with the results in Table 4 of the paper. For Skip RNNs, the mutual information values are in Table 5.

Finally, you can view the results of the permutation test using the script `adaptiveleak/analysis/permutation_test_results.py`. The script takes the following arguments.
```
python permutation_test_results.py --folder <experiment-log> --datasets <list-of-dataset-names>
```
The dataset argument should be a space-separated list of dataset names. The script will print out the fraction (across all budgets) of mutual information values which are significantly different than a randomized association. Section 5.3 in the paper describes this process, as well as the corresponding results across all 9 datasets.

### Attack Classifier
After policy execution, you can execute a more practical attack using a statistical classifier. This model is an AdaBoost ensemble of decision trees that uses the message sizes from consecutive sequences to predict the corresponding event label. To train a classifier for a set of simulator results, navigate into the `adaptiveleak/attack` directory and run the following command.
```
python train.py --policy <policy-name> --encoding <encoding-name> --dataset <dataset-name> --folder <experiment-name> --window-size <window-size> --num-samples <num-samples>
```
For a longer description of each option, run `python train.py --help`. The `folder` option should be the name of the folder containing the experimental logs in `saved_models/<dataset-name>`. These logs are the results of the previous section. Note that the folder name defaults to the current date as described in the section about running experiments.

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
The folder `adaptiveleak/unit_tests` contains two directories of unit tests. These tests execute small portions of the encoding and sampling process. To run the test suite, navigate to the corresponding directory and run the command `python <file-name>.py`. All the tests should pass.

## Hardware (TI MSP430)
The hardware experiments supplement the simulator by executing AGE on a microcontroller. This section requires a TI MSP430 FR5994, as well as a HM-10 BLE module and four jumper wires. To load programs onto the MSP430, you will also need [Code Composer Studio (CCS)](https://software-dl.ti.com/ccs/esd/documents/ccs_downloads.html) from Texas Instruments. The provided implementation was tested on CCSv10. The sections below describe how to configure and run experiments on the MCU.

### Serializing Sampling Policies and Datasets
The folder 'adaptiveleak/msp' contains the MSP430 implementation of all sampling policies and encoding algorithms. This code provides a backbone that features common functionality for all sampling policies. The project uses conditional compilation to customize itself for a given sampling policy and encoding procedure. The script `adaptiveleak/serialize_policy.py` generates a C header file that sets the parameters for a given sampling policy. You can run this script with the following command (must be in the `adaptiveleak` directory).
```
python serialize_policy.py --policy <policy-name> --dataset <dataset-name> --collection-rate <target-fraction> --encoding <encoding-name> --is-msp
```
As usual, running `python serialize_policy.py --help` will provide descriptions of each variable. The output of this script is a file called `policy_parameters.h`. You should copy this file into the `adaptiveleak/msp430` directory. In the paper, we experiment with collection rates `0.4`, `0.7`, and `1.0`.

The experiments use pre-collected datasets, and the code simulates sensing by reading data from the MSP430's FRAM. The script `adaptiveleak/serialize_dataset.py` converts a portion of a pre-collected dataset into a static C array. The MSP430 application then reads from this static array to perform data sampling. You can execute this script within the `adaptiveleak` directory using the command below.
```
python serialize_dataset.py --dataset <dataset-name> --num-seq <num-seq-to-serialize> --offset <seq-offset> --is-msp
```
Running `python serialize_dataset.py --help` will show descriptions of each parameter. The experiments in Section 5.7 use `--num-seq 75` and `--offset 0`. The result of this script is the file `data.h`. You should move this file into the folder `adaptiveleak/msp430`.

#### Aside: Executing Policies C (for debugging)
The above commands prepare the policies and datasets for the TI MSP430. We also provide a standard C implementation which can be executed on normal devices (e.g. a laptop). This implementation is in the folder `adaptiveleak/c_implementation`. To execute this code, you should follow the above steps but **remove the flag `--is-msp` from each command**. You should then copy both the `policy_parameters.h` and the `data.h` files into the `adaptiveleak/c_implementation` folder. You can then compile and execute the code with the following commands.
```
make policy
./policy
```
When the policy is `uniform`, the collection rate is `0.4`, the dataset is `uci_har` and the number of sequences is `75` (offset `0`), the code should print out the following.
```
Collection Rate: 1500 / 3750 (0.400000)
```
The only real use of the C implementation is for debugging parts of the MSP430 implementation on a fully powered system.

### Hardware Setup
Pins for HM-10, remove the jumpers on the MCU for energy measurement

### Running Experiments
Device server, results saved to `device/results` folder.

### Analysis
Run the script in `analysis/analyze_msp_results.py`. The mutual information is `analysis/msp_mutual_information.py`
