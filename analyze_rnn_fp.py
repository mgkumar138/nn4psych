# %%
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import pickle as pk
from tasks import PIE_CP_OB_v2
import matplotlib.pyplot as plt
from torch.nn import init
import utils_calcs, utils_data, ref_info
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d
from copy import deepcopy
import os
from fixed_point_finder.FixedPointFinderTorch import FixedPointFinderTorch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

contexts = ["change-point", "oddball"]  # "change-point","oddball"
num_contexts = len(contexts)
train_cond = True
reward_size = 7.5
max_displacement = 15
max_time = 300
n_trials = 200
epochs = 100

input_dim = 6 + 2  # set this based on your observation space. observation vector is length 4 [helicopter pos, bucket pos, bag pos, bag-bucket pos], context vector is length 2.
hidden_dim = 64  # size of RNN
action_dim = 3  # set this based on your action space. 0 is left, 1 is right, 2 is confirm.

seed = 2025
np.random.seed(seed)
torch.manual_seed(seed)

# weaker top, better bot
model_path = "./model_params/12.0_V3_0.0ns_Nonelb_Noneub_0.7g_64n_40000e_2s.pth"
# noinspection PyPackageRequirements
#model_path = "./model_params/36.0_V3_0.0ns_Nonelb_Noneub_0.95g_64n_40000e_2s.pth"


class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, gain=1.5):
        super(ActorCritic, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gain = gain
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity='tanh')
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        self.init_weights()

    def init_weights(self):
        for name, param in self.rnn.named_parameters():  # initialize the input and rnn weights
            if 'weight_ih' in name:
                # initialize input weights using 1/sqrt(fan_in). if 1/fan_in, more feature learning. 
                init.normal_(param, mean=0, std=1 / (self.input_dim ** 0.5))
            elif 'weight_hh' in name:
                # initialize rnn weights in a stable (gain=1.0) or chaotic regime (gain=1.5)
                init.normal_(param, mean=0, std=self.gain / self.hidden_dim ** 0.5)
            elif 'bias_ih' in name or 'bias_hh' in name:
                # Set RNN biases to zero
                init.constant_(param, 0)

        for layer in [self.actor, self.critic]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    # initialize input weights using 1/fan_in to induce feature learning 
                    init.normal_(param, mean=0, std=1 / self.hidden_dim)
                elif 'bias' in name:
                    init.constant_(param, 0)

    def forward(self, x, hx):
        r, h = self.rnn(x, hx)
        r = r.squeeze(1)
        return self.actor(r), self.critic(r), h

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


model = ActorCritic(input_dim, hidden_dim, action_dim)
if model_path is not None:
    model.load_state_dict(torch.load(model_path))
    print('Load Model')

# print(model.rnn.weight_hh_l0)
# print(model.rnn.bias_hh_l0)
# plt.figure()
# plt.imshow(model.rnn.weight_hh_l0.detach().numpy())
# make_eigenvalue_plot(model.rnn.weight_hh_l0.detach().numpy())
# plt.show()

all_states = np.zeros([epochs, num_contexts, 5, n_trials])
Hs_all = [[],[]]
Os_all = [[],[]]
# get rnn, actor, critic activity
for epoch in range(epochs):
    Hs = []
    As = []
    Cs = []
    Rs = []
    Os = []
    for tt, context in enumerate(contexts):
        env = PIE_CP_OB_v2(condition=context, max_time=max_time, total_trials=n_trials,
                           train_cond=train_cond, max_displacement=max_displacement, reward_size=reward_size)

        h, a, c, r, o = [], [], [], [], []
        hx = torch.randn(1, 1, hidden_dim) * 1 / hidden_dim
        for trial in range(n_trials):

            next_obs, done = env.reset()
            norm_next_obs = env.normalize_states(next_obs)
            next_state = np.concatenate([norm_next_obs, env.context])
            next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

            while not done:
                actor_logits, critic_value, hx = model(next_state, hx)
                probs = Categorical(logits=actor_logits)
                action = probs.sample()

                # Take action and observe reward
                next_obs, reward, done = env.step(action.item())

                h.append(hx[0, 0]), a.append(actor_logits[0]), c.append(critic_value[0]), r.append(reward), o.append(
                    env.hazard_trigger)

                # Prep next state
                norm_next_obs = env.normalize_states(next_obs)
                next_state = np.concatenate([norm_next_obs, env.context])
                next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

        Hs.append(h), As.append(a), Cs.append(c), Rs.append(r), Os.append(o)
        Hs_all[tt].append(torch.stack(h))
        Os_all[tt].append(torch.tensor(o))

        all_states[epoch, tt] = np.array(
            [env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])




# _, _, area = plot_lrs(all_states, scale=0.25)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


def plot_combined_state_space(Hs, Rs, Os, contexts):
    plt.figure(figsize=(12, 6))

    for i, (h_list, r_list, o_list, context) in enumerate(zip(Hs, Rs, Os, contexts)):
        # Convert lists to numpy arrays
        h_array = torch.stack(h_list).detach().numpy()
        r_array = np.array(r_list)
        o_array = np.array(o_list)

        # Perform PCA for dimensionality reduction to 2D
        pca = PCA(n_components=2)
        h_proj = pca.fit_transform(h_array)

        # Calculate differences between consecutive hidden states for vector field
        vectors = h_proj[1:] - h_proj[:-1]

        # Determine marker colors for reward and hazards
        colors = np.where(r_array > 0, 'green', 'gray')  # Default colors based on reward
        # Overlay red color where hazard is indicated (o_array > 0)
        colors[o_array > 0] = 'red'

        # Determine marker sizes based on reward
        sizes = 20  # + ((r_array - 0) / (1 - 0)) * (200 - 20)  # Scale sizes from 20 to 200

        # Plot PCA projection with vector field
        plt.subplot(1, len(contexts), i + 1)

        # Draw paths with a quiver plot
        plt.quiver(h_proj[:-1, 0], h_proj[:-1, 1],
                   vectors[:, 0], vectors[:, 1],
                   angles='xy', scale_units='xy', scale=1, color='purple', alpha=0.7)

        # Scatter plot the hidden states with color coding
        plt.scatter(h_proj[:, 0], h_proj[:, 1], c=colors, s=sizes, alpha=1)

        # Highlight start and end points
        plt.scatter(h_proj[0, 0], h_proj[0, 1], color='blue', s=100, alpha=1, marker='s', label='Start Point')
        plt.scatter(h_proj[-1, 0], h_proj[-1, 1], color='blue', s=100, alpha=1, marker='x', label='End Point')

        plt.scatter([], [], label=context, color='r')
        plt.scatter([], [], label='Bag Drop', color='g')
        plt.scatter([], [], label='Bucket Mvmt', color='gray')

        plt.title(f'Combined State Space - {context}')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()


# Call the combined function with hidden states, rewards, hazard indications, and contexts
#plot_combined_state_space(Hs, Rs, Os, contexts)


context=1
hidden_states = [torch.stack(h).detach().unsqueeze(0).numpy() for h in Hs]
hidden_states_all = [torch.vstack(h).detach().unsqueeze(0).numpy() for h in Hs_all]
stable_fp_cnt, unstable_fp_cnt = find_fixed_points(model, hidden_states_all, context=context, model_name='weaker')
print(stable_fp_cnt, unstable_fp_cnt)
