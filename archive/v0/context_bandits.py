# code to run model
# Ganesh, Adam
#same as 

#%%
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from jax import grad, jit, vmap, random, lax
from jax.nn import softmax, relu, tanh
from jax.nn.initializers import glorot_uniform, normal
from copy import deepcopy

# Define constants
num_epochs = 200
epoch_stop_training = 190
num_contexts = 2
num_trials = 50 # per trial
num_actions = 2
hidden_units = 64
gamma = 0.0  # play around with different gamma between 0.0 to 0.99
seed = 2024
learning_rate = 5e-4

reward_feedback = True
action_feedback = True
context_feedback = False

# Define reward probabilities for each context and each arm
reward_probs = jnp.array([
    [1.0, 0.0],  # Context 1
    [0.0, 1.0],  # Context 2
])

# Initialize model parameters
def initialize_params(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)

    n_input = 1
    if reward_feedback:
        n_input +=1
    if action_feedback:
        n_input +=num_actions
    if context_feedback:
        n_input += num_contexts
    print('Input dimensions: ',n_input)

    Wxh =random.normal(k1, (n_input, hidden_units)) /jnp.sqrt(n_input)
    Whh = random.normal(k2, (hidden_units, hidden_units)) /jnp.sqrt(hidden_units)
    Wha = random.normal(k3, (hidden_units, num_actions)) *1e-3
    Whc = random.normal(k4, (hidden_units, 1)) * 1e-3
    return Wxh, Whh, Wha, Whc

# Recurrent Neural Network forward pass
def rnn_forward(params, inputs, h):
    Wxh, Whh, Wha, Whc = params
    h = tanh(jnp.dot(inputs, Wxh) + jnp.dot(h, Whh))
    return h

# Define policy (actor) and value (critic) functions
def policy_and_value(params, h):
    Wxh, Whh, Wha, Whc = params
    policy = jnp.dot(h, Wha)
    value = jnp.dot(h, Whc) # Critic
    policy_prob = softmax(policy)  # Actor
    return policy_prob, value

def get_onehot_action(policy_prob):
    A = np.random.choice(a=np.arange(num_actions), p=np.array(policy_prob))
    onehotg = np.zeros(num_actions)
    onehotg[A] = 1
    return onehotg

def np_softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max(x) for numerical stability
    return e_x / e_x.sum(axis=-1, keepdims=True)

# Loss function
@jit
def loss_fn(params, state, next_value, prev_h, action, reward):
    # IMPORTANT: Need to re-run these functions within the loss_function to get the gradients with respect to these weights
    h = rnn_forward(params, state, prev_h)
    policy_prob, value = policy_and_value(params, h)

    # compute temporal difference error
    td_errors = reward + gamma * lax.stop_gradient(next_value) - value

    policy_losses = -jnp.log(policy_prob) * action * td_errors
    value_losses = td_errors ** 2

    # combine los
    loss = jnp.mean(policy_losses) + 0.5 * jnp.mean(value_losses)
    return loss

def moving_average(signal, window_size):
    # Pad the signal to handle edges properly
    padded_signal = np.pad(signal, (window_size//2, window_size//2), mode='edge')
    
    # Apply the moving average filter
    weights = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(padded_signal, weights, mode='valid')
    
    return smoothed_signal

def int_to_onehot(index, size):
    onehot_vector = np.zeros(size)
    onehot_vector[index] = 1
    return onehot_vector


# Training loop
def train(params, context, reward_prob,opt_state, prev_h, history, train_var):

    reward = 0.0
    action = np.random.choice([0,1])

    state = np.array([0.0])
    if reward_feedback:
        state = np.concatenate([state, np.array([reward])])
    if context_feedback:
        context_onehot = int_to_onehot(context, num_contexts)
        state = np.concatenate([state, context_onehot])
    if action_feedback:
        action_onehot = int_to_onehot(action, num_actions)
        state = np.concatenate([state, action_onehot])

    params_dict = {'Wxh':[],
                   'Whh':[],
                   'Wha':[],
                   'Whc':[]
                   }
    activity_list = []
    for trial in range(num_trials):
        
        h = rnn_forward(params, state, prev_h)
        policy, _ = policy_and_value(params, h)
        action = get_onehot_action(policy)

        # pass action to env to get next state and reward 
        rprob = reward_prob[np.argmax(action)]
        reward = np.random.choice([0, 1], p=[1 - rprob, rprob])

        # update state (inputs into the RNN)
        next_state = np.array([0.0])
        if reward_feedback:
            next_state = np.concatenate([next_state, np.array([reward])])
        if context_feedback:
            context_onehot = int_to_onehot(context, num_contexts)
            next_state = np.concatenate([next_state, context_onehot])
        if action_feedback:
            next_state = np.concatenate([next_state, action])

        # get next state value prediction
        new_h = rnn_forward(params, next_state, h)
        _, next_value = policy_and_value(params, new_h)

        if train_var:
            # compute the loss with respect to the state, action, reward and newstate
            loss, grads = jax.value_and_grad(loss_fn)(params, state, next_value, prev_h, action, reward)
            #update the weights
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
        else:
            print('no learning')
            loss = 0

        # make sure you assign the state and rnn state correctly for the next trial
        state = next_state
        prev_h = h

        params_dict['Wxh'].append(params[0])
        params_dict['Whh'].append(params[1])
        params_dict['Wha'].append(params[2])
        params_dict['Whc'].append(params[3])
        
        activity_list.append(h)

        history.append([reward, np.argmax(action), loss])

        if trial % 50 == 0:
            print(context, trial, state, reward_prob, np.round(policy,1), reward)

    return params, params_dict, activity_list, history, prev_h, opt_state

#%%
# contextual bandit training
# Initialize parameters & optimizer
params = initialize_params(jax.random.PRNGKey(seed))
initparams = deepcopy(params)
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)
prev_h = random.normal(jax.random.PRNGKey(0), (hidden_units,))*0.1

history = []
store_h = []
store_params = {'Wxh':[],
                   'Whh':[],
                   'Wha':[],
                   'Whc':[]
                   }

# Train the model
for epoch in range(num_epochs):
    if epoch < epoch_stop_training:
        train_var = True
    else:
        train_var = False
    for context in range(num_contexts):
        # depending on the context, determine the reward probabilities

        print(f'### Epoch {epoch} Context {context}')
        reward_prob = reward_probs[context]
        params, params_dict, activity_list, history, prev_h, opt_state = train(params, context, reward_prob, opt_state, prev_h, history,train_var)

        store_h.append(activity_list)
        for i, weight_set in enumerate(['Wxh', 'Whh', 'Wha', 'Whc']):
            store_params[weight_set].append(params_dict[weight_set])

#%% Save store_h, store_params, history

# [note: no training for final 10 epochs x 2 contexts x 50 trials]

np.save('data/activity_contextual.npy', np.array(store_h)) # (400, 50, 64) 
np.save('data/history_contextual.npy', np.array(history)) # (20000, 3)

# weights: Wxh, Whh, Wha, Whc
for i, weight_set in enumerate(['Wxh', 'Whh', 'Wha', 'Whc']):
    np.save(f'data/{weight_set}_contextual.npy', np.array(store_params[weight_set])) # (400, 50, 64, *)

#%%
# Plot the reward over trials
# initial learning
window = 1
view_epochs = 3
initial_learning_trials = view_epochs * num_contexts * num_trials
after_learning_trials = (num_epochs-view_epochs) * num_contexts * num_trials

print(f"Avg rewards before: {np.mean(np.array(history)[:initial_learning_trials,0]):.1f}, after {np.mean(np.array(history)[after_learning_trials:,0]):.1f}")

f,ax = plt.subplots(6,1, figsize=(8,12))
cumr = moving_average(np.array(history)[:initial_learning_trials,0], window_size=window)
ax[0].plot(cumr, zorder=2, color='k')
ax[0].set_xlabel('Trial')
ax[0].set_ylabel('Reward')
ax[0].set_title('Rewards over Trials, Before learning')

cumr = moving_average(np.array(history)[after_learning_trials:,0], window_size=window)
ax[1].plot(cumr, zorder=2, color='k')
ax[1].set_xlabel('Trial')
ax[1].set_ylabel('Reward')
ax[1].set_title('Rewards over Trials, After learning')

# ax[1].plot(np.array(history)[:,2], zorder=2, color='k')
# ax[1].set_xlabel('Trial')
# ax[1].set_ylabel('Loss')
# ax[1].set_title('Actor-Critic Loss over Trials')


ax[2].plot(np.array(history)[:initial_learning_trials,1], zorder=2, color='k')
ax[2].set_xlabel('Trial')
ax[2].set_ylabel('Action')
ax[2].set_title('Actions sampled over contexts, Before learning')

ax[3].plot(np.array(history)[after_learning_trials:,1], zorder=2, color='k')
ax[3].set_xlabel('Trial')
ax[3].set_ylabel('Action')
ax[3].set_title('Actions sampled over contexts, After learning')

ax[4].plot(np.array(history)[:initial_learning_trials,2], zorder=2, color='k')
ax[4].set_xlabel('Trial')
ax[4].set_ylabel('Loss')
ax[4].set_title('Loss over contexts, Before learning')

ax[5].plot(np.array(history)[after_learning_trials:,2], zorder=2, color='k')
ax[5].set_xlabel('Trial')
ax[5].set_ylabel('Loss')
ax[5].set_title('Loss over contexts, After learning')

# ax[3].set_xlabel('Trial')
# ax[3].set_ylabel('Loss')
# ax[3].set_title('Actor-Critic Loss over Trials, After learning')

colors = ['r','b','g', 'y']
for a in range(len(ax)):
    j=1
    for i in range(view_epochs):
        for context in range(num_contexts):
            ax[a].axvline(num_trials*j,color=colors[context], zorder=1)
            j+=1

f.tight_layout()

# save to plots dir
plt.savefig('plots/contextual_bandit.png')


# %%


gammaPlot = 0.5
prediction_errors = []

for t in range(len(history) - 1):
    reward_t = history[t][0]
    reward_t_plus_1 = history[t + 1][0]
    prediction_error = reward_t + gammaPlot * (reward_t_plus_1 - reward_t)
    prediction_errors.append(prediction_error)

f,ax = plt.subplots(5,1, figsize=(8,12))


ax[0].plot(prediction_errors, label='PE', zorder=2, color='k')
ax[0].set_xlabel('Trial')
ax[0].set_ylabel('Reward')
ax[0].set_title('Reward over Trials')

colors = ['r','b','g', 'y']
for a in range(3):
    i = 1
    for epoch in range(num_epochs):
        for context in range(num_contexts):
            ax[a].axvline(num_trials*i,color=colors[context], zorder=1)
            i+=1


inf_vol = np.var(prediction_errors)
window_size = 50
volatility_over_time = np.array([
    np.var(prediction_errors[i:i + window_size]) 
    for i in range(len(prediction_errors) - window_size + 1)
])

ax[1].plot(volatility_over_time, label='PE', zorder=2, color='k')
ax[1].set_xlabel('Trial')
ax[1].set_ylabel('var(reward)')
ax[1].set_title('estimated volatility over time')

# %%
