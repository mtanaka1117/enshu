# Adaptive Group Encoding
[Protecting Adaptive Sampling from Information Leakage on
Low-Power Sensors](https://dl.acm.org/doi/pdf/10.1145/3503222.3507775)
の再現実験手順を記す。

## 再現の手順
### 1. 環境構築
仮想環境を用いる。下記のコマンドで仮想環境を作り、仮想環境内に入る。
```
python3 -m venv adaptiveleak-env
. adaptiveleak-env/bin/activate
```
下記のコマンドを用いてパッケージをインストールする。このコマンドはrootディレクトリで行う。
```
pip3 install --upgrade pip
pip3 install -e .
```

### 2. データセットの入手
[Google Drive](https://drive.google.com/drive/folders/1BrXn-Spc3GwbSmZu-xI5mLefBqNQ8vMa?usp=sharing)からダウンロードする。zipファイルを解凍し、以下のディレクトリにそれぞれ配置する。

1. `datasets/datasets` -> `adaptiveleak/datasets`
2. `saved_models/saved_models` -> `adaptiveleak/saved_models`
3. `traces/traces` -> `adaptiveleak/traces`
<!-- 4. `msp_results.zip` -> `adaptiveleak/device/results` -->

### 3. シミュレーション
#### サンプリング
```
cd adaptiveleak
./run_simulator.sh <dataset-name>
```
結果は`saved_models/<dataset-name>/<date>`に保存される。`<date>`は次の攻撃シミュレーションにおいて使用する。

#### 攻撃シミュレーション
`adaptiveleak/attack`ディレクトリに移動して行う。

```
python train.py --policy <policy-name> --encoding <encoding-name> --dataset <dataset-name> --folder <date> --window-siz/home/mtanaka/adaptive-group-encoding/adaptiveleak/saved_models/eoge <window-size> --num-samples <num-samples>
```

以下の組み合わせを全て実行する

`--policy`：adaptive_deviation, adaptive_heuristic  
`--encoding`：group, standard, padded  
`--dataset`：uci_har, trajectories, eog, haptics, mnist, pavement, tiselac, strawberry, epilepsy  
`--folder`：`<date>`  
`--window-size`：10  
`--num-samples`：10000  


### 4. シミュレーション結果のプロット
`adaptiveleak/analysis`ディレクトリに移動して行う。


論文中のFigure 6を再現するためには、以下を実行する。
```
python plot_all_attacks.py --folder <experiment-name> --datasets uci_har trajectories eog haptics mnist pavement tiselac strawberry epilepsy --output-file [<output-path>]
```

上記を実行した際に
`OSError: 'seaborn-ticks' is not a valid package style, path of style file, URL of style file, or library style name (library styles are listed in `style.available`)`
が出た場合の対応は注意事項に記載。

論文中のTable 4を再現するためには、以下を実行する。
```
python plot_error.py --folder <experiment-name> --dataset <dataset-name> --metric mae --output-file [<output-path>]
```


## 評価環境
Ubuntu(WSL 2)


## 注意事項
### 命名規則
論文と異なる命名規則がある。  
`AGE` -> `group`  
`Linear policy` -> `adaptive heuristic policy`  
`Activity`データセット -> `uci_har`  
`Characters`データセット -> `trajectories`  
`Password`データセット -> `haptic`  

### `'seaborn-ticks' is not a valid package style` errorへの対処
`analysis/plot_utils.py`の`PLOT_STYLE = seaborn-ticks`をコメントアウトし`PLOT_STYLE = 'seaborn-v0_8'`に書き換える。
```
# PLOT_STYLE = 'seaborn-ticks'
PLOT_STYLE = 'seaborn-v0_8'
```



<!-- This repository contains the implementation of Adaptive Group Encoding (AGE), a system for protecting adaptive sampling algorithms from leaking information through communication patterns on low-power devices. This work was accepted into ASPLOS 2022. The repository has the following general structure. Note that most of code supports the simulator framework, and the paths below all lie within the `adaptiveleak` directory.

1. `analysis`: Scripts to analysis experiment results.
2. `attack`: Script to train the attack classifier model.
3. `device`: Holds server code for experiments with the TI MSP430.
4. `energy_systems`: Manages energy traces to track energy consumption in the simulator framework.
5. `msp`: Contains the TI MSP430FR5994 implementation of all encoding and sampling strategies.
6. `plots`: Holds all plots for the included experimental results.
7. `skip_rnn`: Implements a Skip RNN sampling policy. See the README inside this directory for information about how to train Skip RNNs.
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

### Downloading Datasets, Energy Traces, and Saved Results
The datasets, energy traces, and existing results are too large to comfortably fit in the Github repository. You can download this information from [this Google Drive folder](https://drive.google.com/drive/folders/1BrXn-Spc3GwbSmZu-xI5mLefBqNQ8vMa?usp=sharing). Upon downloading each resource, extract the ZIP files. The enclosed directories should be placed at the following locations.

1. `datasets.zip` -> `adaptiveleak/datasets`
2. `saved_models.zip` -> `adaptiveleak/saved_models`
3. `traces.zip` -> `adaptiveleak/traces`
4. `msp_results.zip` -> `adaptiveleak/device/results`

**The exact locations are important, as the code looks for directories in these places.** As a note, the `msp_results` are large (`~3GB`) because they contain raw energy traces over time. If you do not have enough disk space, then you can omit downloading the MSP results. This directory contains logs from the MSP430 device and is not required to run either the simulator or the MSP430 implementation.

### Naming
There are three naming conventions in the code repository that differ from the paper. First, in the code, the `AGE` encoding system is called `group`. Second, the `Linear` policy in the paper is called the adaptive `heuristic` policy in the code. Finally, in the codebase, the `Activity` dataset is called `uci_har`, `Characters` is called `trajectories`, and `Password` is called `haptics`.

### Running Sampling Policies
The entry point for the simulator is the script `adaptiveleak/simulator.py`. You must navigate into the `adaptiveleak` directory to run the script. This code has many command line options which are best viewed when running `python simulator.py --help`. You can execute the simulator on a single dataset and policy using the command below. In the next paragraph, we describe a utility that runs all policies on a single dataset.
```
python simulator.py --dataset <dataset-name> --encoding <encoding-name> --encryption <encryption-type> --collection-rate <budget> --should-print
```
The collection rate is the target fraction of elements in each sequence to capture; the budget is set at the `Uniform` policy's energy consumption at this fraction. You can specify a range of elements by providing three values (space-separated) in the form `<min> <max> <step>`. The results in the paper use `--collection-rate 0.3 1.0 0.1`. As a note, the encoding algorithm `group` is the full `AGE` system. The dataset name is the name of the folder in `datasets` (e.g. `datasets/<dataset-name>`) containing the data files. The shell script `adaptiveleak/run_simulator.sh` executes all policies on the dataset passed as a command line argument (shown below). This script is limited to `standard`, `AGE`, and `Padded` encoding. See below for instructions on how to easily run variants of `AGE`.
```
./run_simlator.sh <dataset-name>
```
This command can take a few minutes to run, especially for the larger datasets (`uci_har`, `mnist`, `tiselac`). The `epilepsy` dataset is relatively small and represents a good starting point. **We include the outputs from all datasets in the folder `adaptiveleak/saved_models/<dataset-name>/results. You may use these logs if it is too time consuming to execute all experiments from scratch.**

After execution, the results are automatically stored in the folder `saved_models/<dataset-name>/<date>`. There will be a folder in this directory for the sampling policy and the encoding algorithm. Keep note of the date, as you will use this value to reference these results during the analysis phase (below).

The codebase supports the ability to evaluate variants of `AGE` which use a selection of features from the full policy. These variants are called `pruned`, `single_group` and `group_unshifted`. Section 5.6 in the paper provides a description of each policy. The script `run_simulator_age_comp.sh` provides the ability to easily run all variants on a dataset. You may run this script using the command below.
```
./run_simulator_age_comp.sh <dataset-name>
```
*You must run this script if you wish to reproduce Table 6 in the paper.*

### Analyzing Experimental Results
The `adaptiveleak/analysis` folder contains a few scripts to process the results of each experiment. This section describes how to compute the reconstruction error, as well as the mutual information between message size and event label.

#### Reconstruction Error
The script `adaptiveleak/analysis/plot_error.py` displays the arithmetic mean reconstruction error for each budget in the executed experiment. You can run this script by navigating to the directory `adaptiveleak/analysis` folder and running the command below.
```
python plot_error.py --folder <experiment-name> --dataset <dataset-name> --metric <metric-name>
```
The arguments are described when running `python plot_error.py --help`. The `--folder` should be the date of the experiment as produced by the execution step in the last section. The code will look for the folder `adaptiveleak/saved_models/<dataset>/<folder>` and retrieve the results from this directory. **If you are referencing the existing results, set `--folder` to `results`**.

The script will produce a plot showing the error for each constraint. The code will also print out the arithmetic mean error (across all constraints) for each policy. When the provided `metric` is `mae`, the printed error values should align with the results in Table 3 of the paper. Note that the plot does not include `padded` policies due to their high error.

By default, the `plot_err.py` script does not include the Skip RNN results, as the Skip RNNs do not operate under the same energy constraints. You can include these error values by including the option `--include-skip-rnn` to the above command. The error results here should align with the MAE values in Table 5.

For brevity, the `plot_error.py` script also does not include the variants of AGE. To perform this analysis, run the above script with the option `--is-group-comp`. The printed result shows the MAE value for each AGE variant. Taking the median of the symmetric percentage error between each variant and AGE from all datasets yields Table 6. The script `analysis/age_comparison.py` performs this computation, and you may run this script using the command below. *Note that you must run the `AGE` variants (e.g. via `run_simulator_age_comp.sh`) to see results for the variant policies.*
```
python age_comparison.py --folder <experiment-name> --datasets <list-of-datasets>
```
The `folder` argument should be the folder containing the experiment results in each dataset. To use the pre-collected results, set `--folder` to `results`. Running the script with the `--help` option provides further descriptions of each argument.

For space reasons, the paper only shows the median percent errors across all datasets and budgets (Table 6). To better verify the results for each variant, the tables below show the average MAE across all budgets on each individual dataset. The below error values should match the results of running the `plot_error.py` script with `--is-group-comp` and a `metric` of `mae`. 

| Dataset | Linear Single | Linear Unshifted | Linear Pruned | Linear AGE |
| ------- | ------------- | ---------------- | ------------- | ---------- |
| Activity | 0.00996 | 0.00992 | 0.02553 | **0.00945** |
| Characters | 0.00467 | 0.00468 | 0.01182 | **0.00463** |
| EOG | 0.12979 | 0.12876 | 0.15106 | **0.12589** |
| Epilepsy | 0.09982 | 0.09973 | 0.15321 | **0.09965** |
| MNIST | 4.96077 | 4.94762 | 5.24250 | **4.93969** |
| Password | 0.00255 | 0.00252 | 0.00317 | **0.00238** |
| Pavement | 0.69418 | 0.70065 | 0.97322 | **0.68862** |
| Strawberry | 0.00507 | 0.00511 | 0.01565 | **0.00501** |
| Tiselac | 6.24563 | 8.99798 | 4.76298 | **2.67698** |

| Dataset | Deviation Single | Deviation Unshifted | Deviation Pruned | Deviation AGE |
| ------- | ------------- | ---------------- | ------------- | ---------- |
| Activity | 0.01090 | 0.01087 | 0.02567 | **0.01041** |
| Characters | 0.00461 | 0.00462 | 0.01187 | **0.00457** |
| EOG | 0.13628 | 0.13493 | 0.14469 | **0.13213** |
| Epilepsy | 0.10080 | 0.10062 | 0.16008 | **0.10050** |
| MNIST | 4.72692 | 4.70312 | 4.98216 | **4.69580** |
| Password | 0.00274 | 0.00270 | 0.00326 | **0.00261** |
| Pavement | 0.68684 | 0.69433 | 1.01491 | **0.67860** |
| Strawberry | 0.00490 | 0.00494 | 0.01334 | **0.00485** |
| Tiselac | 6.32186 | 9.02931 | 4.77306 | **2.79338** |


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
For a longer description of each option, run `python train.py --help`. The `folder` option should be the name of the folder containing the experimental logs in `saved_models/<dataset-name>`. These logs are the results of the previous section. The results in the paper use a `window-size` of `10` and a `num-samples` of `10000` (see Section 5.4 in the paper). It can take a few minutes to run the attack classifier on each dataset.

The training process uses 5-fold stratified cross evaluation. The results get automatically stored in the evaluation logs for each sampling policy, encoding algorithm, and collection rate. Each entry in the serialized result is a list of 5 elements following the 5-fold evaluation. **The existing result logs already contain the attack classifier values.**

The script `analysis/plot_attack.py` analyzes the attack classification results. You can execute this script using the command below.
```
python plot_attack.py --folder <experiment-name> --dataset <dataset-name> --output-file [<optional-output-path>]
```
Executing `python plot_attack.py --help` provides longer descriptions of each argument. The provided `--folder` follows the same convention as the `train.py` script above. To analyze the previous results, use the same `folder`. The `plot_attack.py` script will show the median attack accuracy across all 5 evaluation folds for each energy budget. The script also prints out the median and maximum attack accuracy across all constraints. The maximum values are in parentheses. For the Skip RNN, these values correspond to the Attack results in Table 5 of the paper.

The script `analysis/plot_all_attacks.py` will show the median and maximum attack accuracy values for multiple datasets. You may run this script using the command below.
```
python plot_all_attacks.py --folder <experiment-name> --datasets <list-of-dataset-names> --output-file [<optional-output-path>]
```
The result is a bar chart that shows the median, IQR, and maximum attack accuracy values for each provided dataset. Running this command with all 9 provided datasets yields Figure 6 in the paper. For example, using the existing results, the following command will reproduce Figure 6.
```
python plot_all_attacks.py --folder results --datasets uci_har trajectories eog haptics mnist pavement tiselac strawberry epilepsy
```

The attack logs include confusion matrices for the adversary's classifier. To view the confusion matrices, use the script `adaptiveleak/analysis/view_confusion_mat.py` via the command below. The script will print out the confusion matrix for each of the 5-fold cross-validation runs.
```
python view_confusion_mat.py --log-path <path-to-output-log>
```
To reproduce Figure 7 in the paper, use the command with the log path of `adaptive_heuristic_standard` at an `80%` collection rate as shown below.
```
python view_confusion_mat.py --log-path ../saved_models/epilepsy/results/adaptive_heuristic_standard/adaptive_heuristic-standard-stream-tiny_80.json.gz 
```
We use the second entry in the list of matrices to create Figure 7.

### Unit Tests
The folder `adaptiveleak/unit_tests/utils` contains a suite of unit tests. These tests execute small portions of the encoding and sampling process. To run the tests, navigate to the corresponding directory and run the command `python <file-name>.py`. All the tests should pass.

## Hardware (TI MSP430)
The hardware experiments supplement the simulator by executing AGE on a microcontroller. This section requires a TI MSP430 FR5994 MCU, as well as a HM-10 Bluetooth Low Energy (BLE) module and four jumper wires. To load programs onto the MSP430, you will also need [Code Composer Studio (CCS)](https://software-dl.ti.com/ccs/esd/documents/ccs_downloads.html) from Texas Instruments. The provided implementation was tested on CCS v10.1.0. Finally, to run end-to-end experiments, you will need another computer (e.g. laptop) with BLE capabilities. The sections below describe how to configure and run experiments on the MCU.

*If you do not have the relevant hardware, you can skip to the `Analysis` section below and use the included result logs downloaded from the Google Drive (see section on Downloading).*

### Serializing Sampling Policies and Datasets
The folder 'adaptiveleak/msp' contains the MSP430 implementation of all sampling policies and encoding algorithms. This code provides a backbone that features common functionality for all sampling policies. The project uses conditional compilation to customize itself for a given sampling policy and encoding procedure. The script `adaptiveleak/serialize_policy.py` generates a C header file that sets the parameters for a given sampling policy. You can run this script with the following command (must be in the `adaptiveleak` directory).
```
python serialize_policy.py --policy <policy-name> --dataset <dataset-name> --collection-rate <target-fraction> --encoding <encoding-name> --is-msp
```
As usual, running `python serialize_policy.py --help` will provide descriptions of each variable. The output of this script is a file called `policy_parameters.h`. You should copy this file into the `adaptiveleak/msp430` directory. In the paper, we experiment with collection rates `0.4`, `0.7`, and `1.0` on both the `uci_har` and the `tiselac` tasks.

The experiments use pre-collected datasets, and the code simulates sensing by reading data from the MSP430's FRAM. The script `adaptiveleak/serialize_dataset.py` converts a portion of a pre-collected dataset into a static C array. The MSP430 application then reads from this static array to perform data sampling. You can execute this script within the `adaptiveleak` directory using the command below.
```
python serialize_dataset.py --dataset <dataset-name> --num-seq <num-seq-to-serialize> --offset <seq-offset> --is-msp
```
Running `python serialize_dataset.py --help` will show descriptions of each parameter. The experiments in Section 5.7 use `--num-seq 75` and `--offset 0`. The result of this script is the file `data.h`. You should move this file into the folder `adaptiveleak/msp430`.

#### Optional: Executing Policies C (for debugging)
The above commands prepare the policies and datasets for the TI MSP430. We also provide a standard C implementation which can be executed on normal devices (e.g. a laptop). This implementation is in the folder `adaptiveleak/c_implementation`. To execute this code, you should follow the above steps but **remove the flag `--is-msp` from each command**. You should then copy both the `policy_parameters.h` and the `data.h` files into the `adaptiveleak/c_implementation` folder. You can then compile and execute the code with the following commands.
```
make policy
./policy
```
When the policy is `uniform`, the collection rate is `0.4`, the dataset is `uci_har` and the number of sequences is `75` (offset `0`), the code should print out the following.
```
Collection Rate: 1500 / 3750 (0.400000)
```
The only real use of the C implementation is for debugging aspects of the MSP430 implementation on a fully powered system.

### Hardware and Software Setup
The hardware components consist of a TI MSP430 FR5994 MCU and a HM-10 BLE module. To connect the BLE module to the MCU, you will need four jumper wires. Connect one end of each wire to each of the four pins on the HM-10 module. The four pins are labeled `RX`, `TX`, `VCC` and `GND` (look on the back of the HM-10). You must then connect these HM-10 pins to the MCU's pins in the following manner.
1. `RX` -> `P6.0`
2. `TX` -> `P6.1`
3. `VCC` -> `3V3`
4. `GND` -> `GND`
The application will handle the specifics of interfacing with the BLE module. As a note, to get the same energy readings as present in the paper, you will need to set the advertising interval to as large as possible. The sensor breaks the BLE connection to save energy, and a short advertising interval consumes more energy during the module's sleep mode.

The software component of the MCU is managed through Code Composer Studio (CCS). Inside CCS, create a new empty project and set the device to `FR5994`. Then, copy the contents of `adaptiveleak/msp430` into the CCS project directory. The directory should already contain the target policy and dataset from the serialization steps above. You will have to link the `dsplib` directory to get the project to build. To include the `dsplib`, right click on the project in the project explorer and go to `properties`. Navigate to `CCS Build > MSP430 Compiler > Include Options` and then add a link to `dsplib/include`. The project can take a few seconds to build the first time.

### Running Experiments
The MCU program executes both sampling and encoding on the low-power device. The device sends measurements over a Bluetooth link to a separate server. To execute the end-to-end experiments, you must start the sensor program on the MCU and the server program on the server machine. The sub-sections below discuss these aspects.

#### Launching the Sensor
Code Composer Studio (CCS) has the ability to load and launch programs on the TI MSP430. You may accomplish this by connecting the MSP430 to your computer via USB. You can then load and launch the program using the debug button. Once you start the program, you can kill the debugger. There are two important notes. First, the USB cord also provides the device with power. Unless you have a separate battery pack, you should leave the MCU plugged in during these experiments. Second, program loading requires the RXD, TXD, and SBW jumpers. You should only remove these after launching the application.

#### Launching the Server
The file `adaptiveleak/device/sensor_client.py` contains the server module. Once you navigate to the 'adaptiveleak/device' directory, you can launch the server with the command below. You will need to edit the Python script and change the variable `MAC_ADDRESS` to the MAC address of your HM-10 device.
```
python sensor_client.py --dataset <dataset-name> --policy <policy-name> --collection-rate <collection-rate> --output-folder-name <folder-to-save> --encoding <encoding-name> --trial <trial-num> --max-samples <max-num-seq>
```
The `dataset`, `policy`, and `encoding` parameters should match the sensor configuration. The results from the paper use `75` sequences. Further, for the padded policies, the server uses the `standard` encoding procedure. Executing `python sensor_client.py --help` will show descriptions of each parameter in more detail. You should save all results from a single dataset in the same directory. Each policy should have its own folder with subdirectories for each collection rate. This design yields a structure of `<base-name>/<policy-name>/<collection-rate>`. You can look at the `adaptiveleak/device/results` folder for an example of this structure.

When first launching the program, the script will ask for you `sudo` password. This information is needed to interface with `gatttool`.

The server will print when it has successfully connected to the BLE module. The program will them prompt you to press a button will launch the experiment. This halt provides an opportunity to start measuring the device's energy consumption. You may start this measurement through the TI EnergyTrace tool within Code Composer Studio. **Before starting the experiment, you should start the EnergyTrace tool.** To avoid excess energy consumption, you should remove the jumper wires on MCU's `RXD`, `TXD`, `SBWT`, `5V`, and `J7`. Note that you will need to place these jumpers back on when loading a new program.

Upon completion, the server program saves the error results in the provided output folder within `adaptiveleak/device/results`. The file name is of the form below.
```
<policy-name>_<encoding-name>_<collection-rate>_trial<trial-num>.json.gz
```
After the server finishes, halt the EnergyTrace operation and save the resulting log as a CSV in the same folder as the server results.

#### Reproducing Results from the Paper
The paper contains results of each policy and encoding procedure over the first 75 samples on the `uci_har` and `tiselac` tasks. The policies are `adaptive_deviation`, `adaptive_heuristic`, and `uniform`. We use the `standard`, `padded`, and `group` encoding strategies for both adaptive policies (`group` is the implementation of `AGE`). For the `uniform` policy, we only experiment with `standard` encoding. We run each configuration with the collection rates `0.4`, `0.7`, and `1.0`. To execute each setup, you must first serialize the policy and dataset. Then, copy the resulting files into the CCS project containing the MSP430 implementation. Finally, use the launch the sensor and server to record the policy's behavior.

The process of executing all configurations can be time consuming. To simplify this process, we include results from each setup in the `adaptiveleak/device/results` directory. This folder has the structure `<dataset-name>/<policy-name>/<collection-rate>/<file-name>`.

As a note, we provide an implementation for Skip RNNs on the MCU. In the paper, however, we do not run MCU experiments with Skip RNNs due to the high computational cost of the underlying policy.

### Analysis
The steps below outline the relevant analysis tasks concerning the MCU experiments. The logs from previously-executed experiments are in `adaptiveleak/device/results`. We include results from both the `uci_har` and the `tiselac` datasets.

#### Extract Energy Consumption
The script `adaptiveleak/device/extract_energy.py` reads an EnergyTrace log and synthesizes the energy consumption results. The script will automatically detect the start of the experiment by thresholding the high energy consumption of the Bluetooth module. You can run this script using the following command.
```
python extract_energy.py --folder <folder-with-csvs> --num-seq <num-seq-in-experiment>
```
The script will compute the energy expended on each sequence in the experiment. The results will be saved in a file called `energy.json` within the input folder. If there are multiple trials in the provided folder, the results get merged into one output file.

As a note, the provided results already have the energy values extracted. You do not need to run this step on the pre-collected values.

#### Energy Consumption and Sampling Error
The script `adaptiveleak/analysis/analyze_msp_results.py` computes the sampling error and energy consumption of each policy on the MSP430. You can run this script using the command below (inside the `adaptiveleak/analysis` folder).
```
python analyze_msp_results.py --folder <base-folder-name> --dataset <dataset-name>
```
The `folder` argument should be the base folder name provided to the server script for this dataset. The analysis script will automatically read the results from all policies included in subdirectories. The script calculates the budget based on on the `uniform` policy's energy consumption. Budget violations cause the offending policy to randomly guess sequence elements. To properly compute the random guessing errors (if needed), you must ensure that you first serialize the dataset with enough sequences to cover all trials (see the serialization step above).

The script will print out the results in a Latex table format for each policy. The columns are `collection rate`, `avg error (std)`, `avg energy / seq (std)`. When running the script on the pre-collected results in `adaptiveleak/device/results/uci_har` and `adaptiveleak/device/results/tiselac`, you should get the same results as in Table 8.

#### Mutual Information
The program `adaptiveleak/analysis/msp_mutual_information.py` computes the Normalized Mutual Information (NMI) between message size and event label from the MCU results. After navigating into `adaptiveleak/analysis`, you can run this script using the following command.
```
python msp_mutual_information.py --folder <base-folder-name> --dataset <dataset-name>
```
The arguments should be the same as in the previous step. For each policy, the script will print out the median and maximum mutual information values across all collection rates. The results from the `uci_har` dataset are included in Section `5.7` of the paper. -->
