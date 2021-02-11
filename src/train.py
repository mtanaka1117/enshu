import numpy as np
import h5py
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from policies import Policy, RandomPolicy, AdaptivePolicy
from utils.file_utils import save_pickle_gz


def fit_inference_model(data_file: str, output_file: str):
    # Read the input data
    with h5py.File(data_file, 'r') as fin:
        inputs = fin['inputs'][:]  # [N, T, D]
        output = fin['output'][:]  # [N]

    inputs = inputs.reshape(inputs.shape[0], -1)  # [N, T * D]
    
    scaler = StandardScaler().fit(inputs)
    inputs = scaler.transform(inputs)
    
    clf = MLPClassifier(hidden_layer_sizes=[64], alpha=0.01).fit(inputs, output)

    print('Train Accuracy: {0}'.format(clf.score(inputs, output)))
    
    output_dict = {
        'model': clf,
        'scaler': scaler
    }

    save_pickle_gz(output_dict, output_file)


#def train(data_file: str, policy: Policy, output_file: str):
#    # Read the input data
#    with h5py.File(data_file, 'r') as fin:
#        inputs = fin['inputs'][:]
#        output = fin['output'][:]
#
#    # Fit the data normalization object
#    scaler = StandardScaler()
#    scaler.fit(inputs.reshape(-1, inputs.shape[-1]))
#
#    # Store the measurements as observed on the server
#    server_measurements: List[np.ndarray] = []
#
#    num_samples = inputs.shape[0]
#    for sample_idx in range(num_samples):
#        seq_features = inputs[sample_idx]  # [T, D]
#
#        seq_measurements: List[np.ndarray] = []
#
#        # Reset the policy before every sequence
#        policy.reset()
#
#        num_transmitted = 0
#        for input_features in seq_features:
#            # Update the estimate
#            policy.transition()
#
#            # Determine whether to transmit the measurement
#            measurement = np.expand_dims(input_features, axis=-1)  # [D, 1]
#            num_transmitted += policy.transmit(measurement=measurement)
#
#            # Obtain and scale the estimate
#            estimate = policy.get_estimate().reshape(1, -1)  # [1, D]
#            estimate = scaler.transform(estimate)  # [1, D]
#
#            # Save the estimates for later analysis
#            seq_measurements.append(estimate.reshape(-1))
#
#        # Translate measurements to fixed-point representation
#        transmitted_seq, total_bytes = policy.quantize_seq(seq_measurements,
#                                                           num_transmitted=num_transmitted)
#
#        measurement_array = transmitted_seq.reshape(1, -1)  # [1, T * D]
#        server_measurements.append(measurement_array)
#
#    # Stack inputs
#    model_inputs = np.vstack(server_measurements)  # [N, T * D]
#
#    # Fit the inference model
#    clf = MLPClassifier(hidden_layer_sizes=[64], alpha=0.01, random_state=281, max_iter=500)
#    clf.fit(model_inputs, output)
#
#    # Evaluate the inference model
#    print('Train Accuracy: {0:.5f}'.format(clf.score(model_inputs, output)))
#
#    # Save the results
#    result_dict = {
#        'model': clf,
#        'scaler': scaler
#    }
#    
#    save_pickle_gz(result_dict, output_file)

def train(data_file: str, output_file: str):
    # Read the input data
    with h5py.File(data_file, 'r') as fin:
        inputs = fin['inputs'][:]
        output = fin['output'][:]

    # Fit the data normalization object
    scaler = StandardScaler()
    model_inputs = scaler.fit_transform(inputs.reshape(-1, inputs.shape[-1]))  # [N * T, D]
    model_inputs = model_inputs.reshape(inputs.shape[0], -1)  # [N, T * D]

    # Fit the inference model
    clf = MLPClassifier(hidden_layer_sizes=[64], alpha=0.01, random_state=281, max_iter=500)
    clf.fit(model_inputs, output)

    # Evaluate the inference model
    print('Train Accuracy: {0:.5f}'.format(clf.score(model_inputs, output)))

    # Save the results
    result_dict = {
        'model': clf,
        'scaler': scaler
    }
    
    save_pickle_gz(result_dict, output_file)


if __name__ == '__main__':
    #policy = RandomPolicy(transition_path='transition_model.pkl.gz',
    #                      target=0.7,
    #                      precision=6,
    #                      width=8)

    train(data_file='datasets/uci_har/train/data.h5',
          output_file='saved_models/uci_har/model.pkl.gz')
