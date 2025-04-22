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

# seed = 2025
# np.random.seed(seed)
# torch.manual_seed(seed)

# weaker top, better bot
model_path = "./model_params/12.0_V3_0.0ns_Nonelb_Noneub_0.7g_64n_40000e_2s.pth"
# noinspection PyPackageRequirements
#model_path = "./model_params/36.0_V3_0.0ns_Nonelb_Noneub_0.95g_64n_40000e_2s.pth"

# _, _, area = plot_lrs(all_states, scale=0.25)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Call the combined function with hidden states, rewards, hazard indications, and contexts
#plot_combined_state_space(Hs, Rs, Os, contexts)


context=1
hidden_states = [torch.stack(h).detach().unsqueeze(0).numpy() for h in Hs]
hidden_states_all = [torch.vstack(h).detach().unsqueeze(0).numpy() for h in Hs_all]
stable_fp_cnt, unstable_fp_cnt = find_fixed_points(model, hidden_states_all, context=context, model_name='weaker')
print(stable_fp_cnt, unstable_fp_cnt)
