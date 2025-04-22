#%%
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from tasks import PIE_CP_OB_v2
import matplotlib.pyplot as plt
from torch.nn import init
import utils_calcs, utils_data, ref_info
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d
from copy import deepcopy
import glob

import utils_calcs, utils_data, ref_info
import behav_figures, rnn_model


contexts = ["change-point","oddball"] #"change-point","oddball"
num_contexts = len(contexts)
train_cond = False
reward_size= 5
max_displacement=10
max_time = 300
n_trials = 200
epochs = 100

input_dim = 6+3  # set this based on your observation space. observation vector is length 4 [helicopter pos, bucket pos, bag pos, bag-bucket pos], context vector is length 2.  
hidden_dim = 64  # size of RNN
action_dim = 3  # set this based on your action space. 0 is left, 1 is right, 2 is confirm.

seed = 2025
np.random.seed(seed)
torch.manual_seed(seed)

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


# all_states = np.zeros([epochs, num_contexts, 5, n_trials])

# # get rnn, actor, critic activity
# for epoch in range(epochs):
#     Hs = []
#     As = []
#     Cs = []
#     Rs = []
#     Os = []
#     for tt, context in enumerate(contexts):
#         env = PIE_CP_OB_v2(condition=context, max_time=max_time, total_trials=n_trials, 
#                 train_cond=train_cond, max_displacement=max_displacement, reward_size=reward_size)
        
#         h, a, c, r, o = [],[],[], [], []
#         hx = torch.randn(1, 1, hidden_dim) * 1/hidden_dim
#         for trial in range(n_trials):

#             next_obs, done = env.reset()
#             norm_next_obs = env.normalize_states(next_obs)
#             next_state = np.concatenate([norm_next_obs, env.context, np.array([0.0])])
#             next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

#             while not done:

#                 if np.random.random_sample()< prm:
#                     hx = (torch.randn(1, 1, hidden_dim) * 1/hidden_dim)

#                 actor_logits, critic_value, hx = model(next_state, hx)
#                 probs = Categorical(logits=actor_logits)
#                 action = probs.sample()

#                 # Take action and observe reward
#                 next_obs, reward, done = env.step(action.item())

#                 h.append(hx[0,0]), a.append(actor_logits[0]), c.append(critic_value[0]), r.append(reward), o.append(env.hazard_trigger)

#                 # Prep next state
#                 norm_next_obs = env.normalize_states(next_obs)
#                 next_state = np.concatenate([norm_next_obs, env.context, np.array([reward])])
#                 next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

#         Hs.append(h), As.append(a), Cs.append(c), Rs.append(r), Os.append(o)

#         all_states[epoch, tt] = np.array([env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])




# def get_lrs_v2(states, threshold=20):
#     true_state = states[2]  # bag position
#     predicted_state = states[1]  # bucket position
#     prediction_error = (true_state - predicted_state)[:-1]
#     update = np.diff(predicted_state)

#     idx = prediction_error !=0
#     prediction_error= prediction_error[idx]
#     update = update[idx]
#     learning_rate = update / prediction_error

#     prediction_error = abs(prediction_error)
#     idx = prediction_error>threshold
#     pes = prediction_error[idx]
#     lrs = np.clip(learning_rate,0,1)[idx]

#     sorted_indices = np.argsort(pes)
#     prediction_error_sorted = pes[sorted_indices]
#     learning_rate_sorted = lrs[sorted_indices]

#     return prediction_error_sorted, learning_rate_sorted


# def plot_lrs(states, scale=0.1):
#     epochs = states.shape[0]
#     pess, lrss, area = [],[], []
#     for c in range(2):
#         pes,lrs = [],[]
#         for e in range(epochs):
#             pe, lr = get_lrs_v2(states[e, c])

#             pes.append(pe)
#             lrs.append(lr)

#         pes = np.concatenate(pes)
#         lrs = np.concatenate(lrs)
#         sorted_indices = np.argsort(pes)
#         prediction_error_sorted = pes[sorted_indices]
#         learning_rate_sorted = lrs[sorted_indices]

#         pess.append(prediction_error_sorted)
#         lrss.append(learning_rate_sorted)
#         area.append(np.trapz(learning_rate_sorted, prediction_error_sorted))
    

#     plt.figure(figsize=(3,2.5))
#     colors = ['orange', 'brown']
#     labels = ['CP', 'OB']
#     for i in range(2):
#         window_size = int(len(lrss[i])*scale)
#         smoothed_learning_rate = uniform_filter1d(lrss[i], size=window_size)
#         plt.plot(pess[i], smoothed_learning_rate, color=colors[i], linewidth=2,label=labels[i])
#     plt.legend()
#     plt.xlabel('Prediction error')
#     plt.ylabel('Learning rate')
#     # plt.title(f'CB={area[0]:.1f}, OB={area[1]:.1f}, A={(area[0]-area[1]):.1f}')
#     plt.tight_layout()
#     return pess, lrss, area


# def plot_states(states):
#     contexts = ["Change-point","Oddball"]
#     for c, context in enumerate(contexts):
#         [trials, bucket_positions, bag_positions, helicopter_positions, hazard_triggers] = states[c]

#         plt.figure(figsize=(4, 2.5))
#         # plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='blue')
#         plt.scatter(trials, bag_positions, label='Bag Position', color='red', marker='o', linestyle='-.', alpha=1, edgecolors='k')
#         plt.plot(trials, helicopter_positions, label='Helicopter', color='green', linewidth=3)
#         plt.plot(trials, bucket_positions, label='Bucket Position', color='orange', alpha=1, linewidth=3)

#         plt.ylim(-10, 310)  # Set y-axis limit from 0 to 300
#         plt.xlabel('Trial')
#         plt.ylabel('Position')
#         plt.title(f"{context}\n$\gamma={gamma}, \\beta_\delta={tds}, p_{{reset}}={prm}, t_{{rollout}}={troll}$")
#         plt.legend(frameon=True, fontsize=8)
#         plt.tight_layout()
#         plt.savefig(f'./analysis/{context}_states.png')
#         plt.savefig(f'./analysis/{context}_states.svg')



def plot_combined_state_space(Hs, Rs, Os):
    '''
    -Takes in RNN weights and returns a plot of the state space
    '''
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np

    contexts = ["Change-point","Oddball"]
    plt.figure(figsize=(4, 8))
    
    for i, (h_list, r_list, o_list, context) in enumerate(zip(Hs, Rs, Os, contexts)):
        # Convert lists to numpy arrays
        h_array = torch.stack(h_list).detach().numpy()
        r_array = np.array(r_list)
        o_array = np.array(o_list)

        # Perform PCA for dimensionality reduction to 2D
        pca = PCA(n_components=2)
        h_proj = pca.fit_transform(h_array)
        pc1, pc2 = pca.explained_variance_ratio_

        # Calculate differences between consecutive hidden states for vector field
        vectors = h_proj[1:] - h_proj[:-1]

        # Determine marker colors for reward and hazards
        colors = np.where(r_array > 0, 'green', 'gray') # Default colors based on reward
        # Overlay red color where hazard is indicated (o_array > 0)
        colors[o_array > 0] = 'red'

        # Determine marker sizes based on reward
        sizes = 20# + ((r_array - 0) / (1 - 0)) * (200 - 20)  # Scale sizes from 20 to 200

        # Plot PCA projection with vector field
        plt.subplot(len(contexts),1, i+1)
        
        # Draw paths with a quiver plot
        plt.quiver(h_proj[:-1, 0], h_proj[:-1, 1],
                   vectors[:, 0], vectors[:, 1],
                   angles='xy', scale_units='xy', scale=1, color='purple', alpha=0.7)
        
        # Scatter plot the hidden states with color coding
        plt.scatter(h_proj[:, 0], h_proj[:, 1], c=colors, s=sizes, alpha=1)
        
        # Highlight start and end points
        plt.scatter(h_proj[0, 0], h_proj[0, 1], color='blue', s=100, alpha=1, marker='s', label='Start Point')
        plt.scatter(h_proj[-1, 0], h_proj[-1, 1], color='blue', s=100, alpha=1, marker='x', label='End Point')
        
        plt.scatter([],[], label=context, color='r')
        plt.scatter([],[], label='Bag Drop', color='g')
        plt.scatter([],[], label='Bucket Mvmt', color='gray')

        plt.title(context)
        plt.suptitle(f'$\gamma={gamma}, \\beta_\delta={tds}, p_{{reset}}={prm}, t_{{mem}}={troll}$')
        plt.title(f"{context}")
        plt.xlabel(f'PC 1 (Var={pc1:.3f})')
        plt.ylabel(f'PC 2, (Var={pc2:.3f})')
        plt.grid(True)
        if i>0:
            plt.legend()

    plt.tight_layout()
    plt.show()


_, rnn_activity = rnn_model.run_rnn(model_path, epochs=epochs, reset_memory=prm)


# _,_,area = plot_lrs(all_states,scale=0.05)
utils_calcs.plot_lrs(all_states,scale=0.05)

# for e in range(5):
#     plot_states(all_states[e])

for e in range(5): 
    utils_calcs.plot_states(all_states[e])

# Call the combined function with hidden states, rewards, hazard indications, and contexts
# plot_combined_state_space(Hs, Rs, Os)