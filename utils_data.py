#%%
'''
Useful functions 
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d
from scipy import stats

def extract_states(states):
    '''
    out of date - use extract_states_v2 instead
    '''
    # originally by Adam
    # Extract prediction error (PE) and state (s) and predicted state (s_hat)


    true_state = states[2]  # bag position
    predicted_state = states[1]  # bucket position
    prediction_error = abs(true_state - predicted_state)
    prediction_error = np.minimum(prediction_error, 100)
    prediction_error = prediction_error[:-1] 

    update = abs(np.diff(predicted_state))
    learning_rate = np.where(prediction_error != 0, update / prediction_error, 0)

    hazard_trials = states[4]
    hazard_indexes = np.where(states[4] == 1)[0]
    hazard_distance = np.zeros(len(states[0]), dtype=int)
    current = 0
    for i in range(len(states[0])):
        if i in hazard_indexes:
            current = 0
        hazard_distance[i] = current
        current += 1
    return prediction_error, update, learning_rate, true_state, predicted_state,hazard_distance, hazard_trials


def unpickle_state_vector(file_dir:str = "data/rnn_behav/model_params_101000/", RNN_param: str="None"):
    """
    Unpickle the state vector made by get_behavior.py.
    
    Parameters:
        state_vector (str): Path to the state vector file.
    
    Returns:
        numpy.ndarray: Unpickled state vector.
    """
    import os
    import pickle

    #available RNN params = "gamma", "preset", "rollout", "scale", "combined"

    with open(os.path.join(file_dir, f"{RNN_param}_cp_list.pkl"), 'rb') as f:
        cp_array = pickle.load(f)

    with open(os.path.join(file_dir, f"{RNN_param}_ob_list.pkl"), 'rb') as f:
        ob_array = pickle.load(f)

    with open(os.path.join(file_dir, f"{RNN_param}_dict.pkl"), 'rb') as f:
        model_list = pickle.load(f)

    # # each array has: [trials, bucket_positions, bag_positions, helicopter_positions, hazard_triggers]
    # agent_list_cp = [[x[1], x[2], 'changepoint'] for x in cp_array]
    # agent_list_ob = [[x[1], x[2], 'oddball'] for x in ob_array]

    return cp_array, ob_array, model_list

def filter_data(data_dir = "./model_params_101000/", threshold = 10):
    '''
    returns - index of models that meet the performance filter 
    '''
    gamma_idx = {}
    rollout_idx = {}
    preset_idx = {}
    scale_idx = {}

    #all possible hyper parameters
    gammas  = [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.25, 0.1]
    rollouts = [5, 10, 20, 30, 50, 75, 100, 150, 200]  # skipped 40
    presets = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    scales  = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

    # Define a dictionary of hyperparameter configurations (value list and file pattern)
    param_configs = {
        "gamma": (
            gammas,
            "*_V3_{val}g_0.0rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        ),
        "rollout": (
            rollouts,
            "*_V3_0.95g_0.0rm_{val}bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        ),
        "preset": (
            presets,
            "*_V3_0.95g_{val}rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        ),
        "scale": (
            scales,
            "*_V3_0.95g_0.0rm_100bz_0.0td_{val}tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        )
    }

    for param_type, (values, pattern) in param_configs.items():
        for val in values:
            file_names = data_dir + pattern.format(val=val)
            models = glob.glob(file_names)
            # Initial cutoff: only keep models with performance metric > 5 (previously done)
            initial_filtered = [m for m in models if float(m.split("\\")[-1].split("_")[0]) > 5]
            # Create a boolean index array based on the second cutoff (performance > threshold)
            idx = [float(m.split("\\")[-1].split("_")[0]) > threshold for m in initial_filtered]
            if param_type == "gamma":
                gamma_idx[val] = idx
            elif param_type == "rollout":
                rollout_idx[val] = idx
            elif param_type == "preset":
                preset_idx[val] = idx
            elif param_type == "scale":
                scale_idx[val] = idx

    return gamma_idx, rollout_idx, preset_idx, scale_idx

def saveload(filename, variable, opt):
    import pickle
    if opt == 'save':
        with open(f"{filename}.pickle", "wb") as file:
            pickle.dump(variable, file)
        print('file saved')
    else:
        with open(f"{filename}.pickle", "rb") as file:
            return pickle.load(file)

