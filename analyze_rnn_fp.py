# %%
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import pickle as pk
from tasks import PIE_CP_OB_v2
import matplotlib.pyplot as plt
from torch.nn import init
import utils_calcs, utils_data, config
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d
from copy import deepcopy
import os
from fixed_point_finder.FixedPointFinderTorch import FixedPointFinderTorch


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


seed = 2025
np.random.seed(seed)
torch.manual_seed(seed)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#import ref_data
#setup model path
#setup model

# contexts = ["change-point", "oddball"]  # "change-point","oddball"
# num_contexts = len(contexts)
# train_cond = True
# reward_size = 7.5
# max_displacement = 15
# max_time = 300
# n_trials = 200
# epochs = 100

# input_dim = 6 + 2  # set this based on your observation space. observation vector is length 4 [helicopter pos, bucket pos, bag pos, bag-bucket pos], context vector is length 2.
# hidden_dim = 64  # size of RNN
# action_dim = 3  # set this based on your action space. 0 is left, 1 is right, 2 is confirm.


# weaker top, better bot
# model_path = "./model_params/12.0_V3_0.0ns_Nonelb_Noneub_0.7g_64n_40000e_2s.pth"
# noinspection PyPackageRequirements
#model_path = "./model_params/36.0_V3_0.0ns_Nonelb_Noneub_0.95g_64n_40000e_2s.pth"

# _, _, area = plot_lrs(all_states, scale=0.25)


# Call the combined function with hidden states, rewards, hazard indications, and contexts
#plot_combined_state_space(Hs, Rs, Os, contexts)


def find_fixed_points(model, hidden_states, context=0, load_from_file = True, model_name=''):

    state_traj = hidden_states[context]
    NOISE_SCALE = 1.0  # Standard deviation of noise added to initial states
    N_INITS = state_traj.shape[1]  # The number of initial states to provide
    noise = np.random.randn(*state_traj.shape)
    noisy_state_traj = state_traj + noise
    #state_traj += noise
    '''Fixed point finder hyperparameters. See FixedPointFinder.py for detailed
	descriptions of available hyperparameters.'''
    fpf_hps = {
        'max_iters': 20000,
        'lr_init': 1.,
        'outlier_distance_scale': 100.0,
        'verbose': True,
        'super_verbose': True}

    # Setup the fixed point finder
    fpf = FixedPointFinderTorch(model.rnn, **fpf_hps)

    # initial_states = fpf.sample_states(hidden_states,
    # 	n_inits=N_INITS,
    # 	noise_scale=NOISE_SCALE)
    initial_states = state_traj[0]

    # Study the system in the absence of input pulses (e.g., all inputs are 0 except the context cue)
    #inputs = np.zeros([1, model.input_dim])
    inputs = np.zeros((N_INITS, model.input_dim))
    inputs[:, -2+context] = 1.0
    # Run the fixed point finder
    fp_fname = './saved_fp/unique_fps_context_{}_model_{}.pk'.format(context, model_name)
    if os.path.exists(fp_fname) and load_from_file:
        with open(fp_fname, 'rb') as f:
            unique_fps = pk.load(f)
    else:
        unique_fps, all_fps = fpf.find_fixed_points(noisy_state_traj[0].copy(), inputs)
        with open(fp_fname, 'wb') as f:
            pk.dump(unique_fps, f)

    # Visualize identified fixed points with overlaid RNN state trajectories
    # All visualized in the 3D PCA space fit the the example RNN states.
    stable_fp_cnt = 0
    unstable_fp_cnt = 0
    for i, fp in enumerate(unique_fps):
        e_vals = fp.eigval_J_xstar[0]
        is_stable = np.all(np.abs(e_vals) < 1.0)
        if is_stable:
            stable_fp_cnt += 1
        else:
            unstable_fp_cnt += 1
    return stable_fp_cnt, unstable_fp_cnt

def analyze_fixed_points(model, rnn_act_dict, model_name=''):

    # Get out Hs, Hs_all from rnn_act_dict
    Hs = rnn_act_dict['Hs']
    Hs_all = rnn_act_dict['Hs_all']

    context = 1
    hidden_states = [torch.stack(h).detach().unsqueeze(0).numpy() for h in Hs]
    hidden_states_all = [torch.vstack(h).detach().unsqueeze(0).numpy() for h in Hs_all]
    stable_fp_cnt, unstable_fp_cnt = find_fixed_points(model, hidden_states_all, context=context, model_name=model_name)
    print(f"Model: {model_name}, Stable Fixed Points: {stable_fp_cnt}, Unstable Fixed Points: {unstable_fp_cnt}")
    return stable_fp_cnt, unstable_fp_cnt
