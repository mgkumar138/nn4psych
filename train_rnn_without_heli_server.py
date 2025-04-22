#%%
'''
Idea is to pretrain a vanilla RNN using RL on several epochs of the helicopter task 
so that it sort of knows what to do. 
The model weights are saved to run a single epoch of 100 trials of each condition, 
similar to Nassar et al. 2021 
and for additional analyses
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, required=False, help='epochs', default=2)
parser.add_argument('--trials', type=int, required=False, help='trials', default=200)

parser.add_argument('--maxdisp', type=int, required=False, help='maxdisp', default=20)
parser.add_argument('--rewardsize', type=int, required=False, help='rewardsize', default=10)

parser.add_argument('--lr', type=float, required=False, help='lr', default=0.0001)
parser.add_argument('--gamma', type=float, required=False, help='gamma', default=0.9)
parser.add_argument('--nrnn', type=int, required=False, help='nrnn', default=64)
parser.add_argument('--loadmodel',type=int, required=False, help='loadmodel', default=0)

parser.add_argument('--seed', type=int, required=False, help='seed', default=0)

args, unknown = parser.parse_known_args()
print(args)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from tasks import PIE_CP_OB
import matplotlib.pyplot as plt
from torch.nn import init
from plots_behav import plot_analysis, get_lrs, saveload
from scipy.stats import linregress
# Assuming that PIE_CP_OB is a gym-like environment
# from your_environment_file import PIE_CP_OB

# Env parameters
n_epochs = args.epochs  # number of epochs to train the model on. Similar to the number of times the agent is trained on the helicopter task. 
n_trials = args.trials  # number of trials per epoch for each condition.
max_time = 300  # number of time steps available for each trial. After max_time, the bag is dropped and the next trial begins after.

train_epochs = 0 #n_epochs*0.5  # number of epochs where the helicopter is shown to the agent. if 0, helicopter is never shown.
no_train_epochs = []  # epoch in which the agent weights are not updated using gradient descent. To see if the model can use its dynamics to solve the task instead.
contexts = ["change-point","oddball"] #"change-point","oddball"
num_contexts = len(contexts)

# Task parameters
max_displacement = args.maxdisp # number of units each left or right moves.
step_cost = 0 #-1/300  # penalize every additional step that the agent does not confirm. 
reward_size = args.rewardsize # smaller value means a tighter margin to get reward.
alpha = 1

# Model Parameters
input_dim = 4+2  # set this based on your observation space. observation vector is length 4 [helicopter pos, bucket pos, bag pos, bag-bucket pos], context vector is length 2.  
hidden_dim = args.nrnn  # size of RNN
action_dim = 3  # set this based on your action space. 0 is left, 1 is right, 2 is confirm.
learning_rate = args.lr
gamma = args.gamma
reset_memory = 200  # reset RNN activity after T trials
bias = [0, 0, -1]
beta_ent = 0.0
seed = args.seed
loadmodel = args.loadmodel
exptname = f"noheli_{loadmodel}pre_{max_displacement}md_{reward_size}rs_{hidden_dim}n_{learning_rate}lr_{n_epochs}e_{seed}s"
print(exptname)

if loadmodel == 1:
    model_path = f'./model_params/pre_model_params_{max_displacement}_heliTrue.pth'


# Actor-Critic Network with RNN
class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, gain=1.0):
        super(ActorCritic, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gain =gain
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity='tanh')
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def init_weights(self):
        for name, param in self.rnn.named_parameters(): # initialize the input and rnn weights 
            if 'weight_ih' in name:
                # initialize input weights using 1/sqrt(fan_in). if 1/fan_in, more feature learning. 
                init.normal_(param, mean=0, std=1/(self.input_dim**0.5 * self.hidden_dim))
            elif 'weight_hh' in name:
                # initialize rnn weights in a stable (gain=1.0) or chaotic regime (gain=1.5)
                init.normal_(param, mean=0, std=self.gain / self.hidden_dim**0.5)
        
        for layer in [self.actor, self.critic]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    # initialize input weights using 1/fan_in to induce feature learning 
                    init.normal_(param, mean=0, std=1/self.hidden_dim)
                elif 'bias' in name:
                    init.constant_(param, 0)

    def forward(self, x, hx):
        r, h = self.rnn(x, hx)
        r = r.squeeze(1)
        return self.actor(r), self.critic(r), h
    
def to_numpy(tensor):
    return tensor.cpu().detach().numpy()



def train(env, model, optimizer,epoch, n_trials, gamma):

    totG = []
    totloss = []
    tottime = []
    hx = torch.randn(1, 1, hidden_dim) *0.001  # initialize RNN activity with random

    for trial in range(n_trials):
        obs, done = env.reset()  # reset env at the start of every trial to change helocopter pos based on hazard rate

        norm_obs = env.normalize_states(obs)  # normalize vector to bound between something resonable for the RNN to handle

        state = np.concatenate([norm_obs,env.context])
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # add batch and seq dim

        # Detach hx to prevent backpropagation through the entire history
        hx = hx.detach()
        if trial % reset_memory == 0:
            # Initialize the RNN hidden state
            hx = torch.randn(1, 1, hidden_dim) *0.001

        log_probs = []
        values = []
        rewards = []
        entropies = []
        totR = 0 

        while not done: #allows multiple actions in one trial (incrementally moving bag_position)
            # Forward pass
            actor_logits, critic_value, hx = model(state, hx)

            bias_tensor = torch.tensor(bias, dtype=actor_logits.dtype, device=actor_logits.device)
            actor_logits += bias_tensor

            probs = Categorical(logits=actor_logits)
            action = probs.sample()

            log_probs.append(probs.log_prob(action))
            values.append(critic_value)

            entropies.append(probs.entropy())

            next_obs, reward, done = env.step(action.item())
            norm_next_obs = env.normalize_states(next_obs)
            next_state = np.concatenate([norm_next_obs,env.context])
            rewards.append(reward)
            totR += reward
            state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

            if done:
                break
        
        # print("trial:", env.trial, "time:", env.time, "obs:", obs, "actor:", actor_logits, "action:", action, "reward:", reward, "next_obs:", next_obs)

        # Calculate returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze()

        # Compute advantages
        advantages = returns - values.detach()

        # Calculate loss
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = ((returns - values)**2).mean()
        entropy_loss = -torch.stack(entropies).mean()

        loss = actor_loss + 0.5 * critic_loss + beta_ent * entropy_loss

        #  train network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totG.append(totR)
        totloss.append(to_numpy(loss))
        tottime.append(env.time)

    return np.array(totG), np.array(totloss), np.array(tottime)


model = ActorCritic(input_dim, hidden_dim, action_dim)
if loadmodel:
    print('Load Model')
    model.load_state_dict(torch.load(model_path))

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epoch_G = np.zeros([n_epochs, num_contexts, n_trials])
epoch_loss = np.zeros([n_epochs, num_contexts, n_trials])
epoch_time = np.zeros([n_epochs, num_contexts, n_trials])
all_states = np.zeros([n_epochs, num_contexts, 5, n_trials])
all_lrs = np.zeros([n_epochs, num_contexts, n_trials-1])
all_pes = np.zeros([n_epochs, num_contexts, n_trials-1])

for epoch in range(n_epochs):

    if epoch < train_epochs:
        train_cond = True  # give helicopter position for these epochs to simulate train condition
    else:
        train_cond = False # dont give helicopter position for these epochs to simulate test condition

    idxs = np.random.choice(np.arange(2), size=2, replace=False)  # randomize CP and OB conditions

    for tt in idxs:

        task_type = contexts[tt]

        env = PIE_CP_OB(condition=task_type, max_time=max_time, total_trials=n_trials, 
                        train_cond=train_cond, max_displacement=max_displacement, reward_size=reward_size, step_cost=step_cost, alpha=alpha)

        totG, totloss,tottime = train(env, model, optimizer, epoch=epoch, n_trials=n_trials, gamma=gamma)

        epoch_G[epoch, tt] = totG
        epoch_loss[epoch, tt] = totloss
        epoch_time[epoch, tt] = tottime

        all_states[epoch, tt] = np.array([env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])
        all_pes[epoch,tt], all_lrs[epoch, tt] = get_lrs(all_states[epoch, tt])

        print(f"Epoch {epoch}, Task {task_type}, G {np.mean(totG):.3f}, L {np.mean(totloss):.3f}, t {np.mean(tottime):.3f}, lr {np.sum(all_lrs[epoch,tt]):.3f}")

# # Calculate the difference in learning rates between CP and OB conditions. Should be positive. 
# cp_vs_ob = plot_analysis(epoch_G, epoch_loss, epoch_time, all_states[-1])

# print(cp_vs_ob)

# plt.figure(figsize=(3,2))
# cp_ob = np.sum(all_lrs[:,0]-all_lrs[:,1],axis=1)
# slope, intercept, r_value, p_value, std_err = linregress(np.arange(n_epochs), cp_ob)
# plt.plot(cp_ob)
# plt.plot(np.sum(all_lrs,axis=2))
# plt.plot(slope * np.arange(n_epochs) + intercept, color='k', label=f'm={slope:.3f}, c={intercept:.2f}, r={r_value:.3f}, p={p_value:.3f}')
# plt.xlabel('Epoch')
# plt.ylabel('CP vs OB') # should become more positive.
# plt.legend()

np.savez(f'./state_data/{exptname}.npz', all_states, epoch_G, epoch_loss, epoch_time)
print('Save States')
print(exptname)