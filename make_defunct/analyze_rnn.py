#%%
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from tasks import PIE_CP_OB_v2
import matplotlib.pyplot as plt
from torch.nn import init
import utils_calcs, utils_data, config
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d
from copy import deepcopy
import glob

import utils_calcs, utils_data, config
import plots_behav, model_rnn


# contexts = ["change-point","oddball"] #"change-point","oddball"
# num_contexts = len(contexts)
# train_cond = False
# reward_size= 5
# max_displacement=10
# max_time = 300
# n_trials = 200
# epochs = 100

# input_dim = 6+3  # set this based on your observation space. observation vector is length 4 [helicopter pos, bucket pos, bag pos, bag-bucket pos], context vector is length 2.  
# hidden_dim = 64  # size of RNN
# action_dim = 3  # set this based on your action space. 0 is left, 1 is right, 2 is confirm.

# seed = 2025
# np.random.seed(seed)
# torch.manual_seed(seed)

# model_path = "./model_params/36.0_V3_0.0ns_Nonelb_Noneub_0.95g_64n_40000e_2s.pth" # good model
# model_path = "./model_params_gamma/12.0_V3_0.0ns_Nonelb_Noneub_0.7g_64n_40000e_2s.pth" # subptimal

gamma = 0.95
tds = 0.25
prm = 0.0
troll = 100
idx = -3

models = f"./model_params_101000/*_V3_{gamma}g_{prm}rm_{troll}bz_0.0td_{tds}tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
files = glob.glob(models)
sorted_file_paths = sorted(files, key=lambda x: float(x.split('/')[2].split('_')[0]))
model_path = sorted_file_paths[idx]
print(model_path)


model = ActorCritic(input_dim, hidden_dim, action_dim, noise=0.0)
if model_path is not None:
    model.load_state_dict(torch.load(model_path))
    print('Load Model')
else: 
    raise FileNotFoundError('Model path not found')

    


_, rnn_activity = model_rnn.run_rnn(model_path, epochs=epochs, reset_memory=prm)


# _,_,area = plot_lrs(all_states,scale=0.05)
utils_calcs.plot_lrs(all_states,scale=0.05)

# for e in range(5):
#     plot_states(all_states[e])

for e in range(5): 
    utils_calcs.plot_states(all_states[e])

# Call the combined function with hidden states, rewards, hazard indications, and contexts
plot_combined_state_space(Hs, Rs, Os)

