# code to run model
# Ganesh, Adam

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
from tasks import ContinuousPredictiveInferenceEnv

# Define constants
num_epochs = 500
num_trials = 200
epoch_stop_training = 490
num_context = 1
obs_size = 3
num_actions = 2
hidden_units = 64
gamma = 0.95  # play around with different gamma between 0.0 to 0.99
seed = 2024
learning_rate = 5e-4

reward_feedback = False
action_feedback = False

# Initialize model parameters
def initialize_params(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)

    n_input = obs_size+num_context
    if reward_feedback:
        n_input +=1
    if action_feedback:
        n_input +=num_actions

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

    # combine loss
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
def train(params, task_type,opt_state, prev_h, history, train_var):

    env = ContinuousPredictiveInferenceEnv(condition=task_type) #DiscretePredictiveInferenceEnv(condition=task_type)
    
    done = False
    reward = 0.0
    total_reward = 0
    action = np.random.choice([0,1])
    
    store_states = []
    store_h = []

    obs = env.reset()
    state = np.array(obs/300)
    if task_type == "change-point":
        context =  np.array([0])
    elif task_type == "oddball":
        context =  np.array([1])
    state = np.concatenate([state,context])


    if reward_feedback:
        state = np.concatenate([state, np.array([reward])])
    if action_feedback:
        action_onehot = int_to_onehot(action, num_actions)
        state = np.concatenate([state, action_onehot])

    while not done:
        
        h = rnn_forward(params, state, prev_h)
        policy, _ = policy_and_value(params, h)
        action = get_onehot_action(policy)

        next_obs, unnorm_reward, done, _ = env.step(np.argmax(action))
        next_state = next_obs/300
        next_state = np.concatenate([next_state,context])
        reward = (unnorm_reward/300)
        total_reward += reward

        if reward_feedback:
            next_state = np.concatenate([next_state, np.array([reward])])
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

        # if env.trial % 100 == 0:
        #     print(np.round(state,1),np.argmax(action), reward)
        
        history.append([reward, np.argmax(action), loss])
        store_states.append(state)
        store_h.append(h)

        # make sure you assign the state and rnn state correctly for the next trial
        state = next_state
        prev_h = h

    return params, history, prev_h, opt_state, total_reward, store_states, store_h

#%%
# Helicopter task
# Initialize parameters & optimizer
params = initialize_params(jax.random.PRNGKey(seed))
initparams = deepcopy(params)
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)
prev_h = random.normal(jax.random.PRNGKey(seed), (hidden_units,))*0.1

history = []
store_rnn = []
store_params = []
store_states = []

# Train the model
for epoch in range(num_epochs):
    if epoch < epoch_stop_training:
        train_var = True
    else:
        train_var = False

    for task_type in ["change-point"]:

        params, history, prev_h, opt_state, total_reward, store_s, store_h = train(params, task_type, opt_state, prev_h, history,train_var)

        print(f'### Epoch {epoch}, R: {total_reward}')

        # store_h.append(prev_h)
        store_params.append(params)
        store_states.append(store_s)
        store_rnn.append(store_h)

np.save('data/activity_helicopter.npy', np.array(store_rnn)) # (400, 50, 64) 
np.save('data/history_helicopter.npy', np.array(history)) # (20000, 3)

#%%
window = 1
view_epochs = 1
initial_learning_trials = view_epochs * num_trials
after_learning_trials = (num_epochs-view_epochs) * num_trials

print(f"Avg rewards before: {np.mean(np.array(history)[:initial_learning_trials,0]):.1f}, after {np.mean(np.array(history)[after_learning_trials:,0]):.1f}")

f,ax = plt.subplots(8,1, figsize=(8,15))
cumr = moving_average(np.array(history)[:initial_learning_trials,0], window_size=window)
ax[0].plot(cumr, zorder=2, color='k')
ax[0].set_xlabel('Trial')
ax[0].set_ylabel('Error')
ax[0].set_title('Rewards over Trials, Before learning')

cumr = moving_average(np.array(history)[after_learning_trials:,0], window_size=window)
ax[1].plot(cumr, zorder=2, color='k')
ax[1].set_xlabel('Trial')
ax[1].set_ylabel('Error')
ax[1].set_title('Rewards over Trials, After learning')

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

before_states = np.array(store_states[0])
ax[6].plot(before_states[:,0], zorder=2, color='g', label='Model')
ax[6].scatter(np.arange(200), before_states[:,1], zorder=1, color='r', label='Ball')
ax[6].set_xlabel('Trial')
ax[6].set_ylabel('Location')
ax[6].set_title('Before learning')
ax[6].legend()

after_states = np.array(store_states[-1])
ax[7].plot(after_states[:,0], zorder=2, color='g', label='Model')
ax[7].scatter(np.arange(200),after_states[:,1], zorder=1, color='r', label='Ball')
ax[7].set_xlabel('Trial')
ax[7].set_ylabel('Location')
ax[7].set_title('After learning')
ax[7].legend()

f.tight_layout()

#%%
# save
import pickle
with open("heli_trained_rnn.pkl", "wb") as f:
    pickle.dump((store_params, store_rnn, store_states, history), f)

#%%
from scipy.optimize import curve_fit
plt.figure(figsize=(3, 3))
labels = ['Before', 'After']
colors=['b', 'r']

# Define the logistic function
def logistic_function(x, L ,x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

# save trained model
t=0
x = np.array(store_states[0])[:,2]
y = np.array(history)[:200,1]

initial_guess = [1, 5, 1, 0]

# Fit the curve
params, covariance = curve_fit(logistic_function, x, y, p0=initial_guess)
x_fit = np.linspace(np.min(x), np.max(x), 500)
y_fit = logistic_function(x_fit, *params)

plt.scatter(x, y, color=colors[t])
plt.plot(x_fit, y_fit, color=colors[t], label=labels[t])

t=1
x = np.array(store_states[-1])[:,2]
y = np.array(history)[-200:,1]

# Initial guess for the parameters: L, x0, k, and b
initial_guess = [1, 5, 1, 0]

# Fit the curve
params, covariance = curve_fit(logistic_function, x, y, p0=initial_guess)
x_fit = np.linspace(np.min(x), np.max(x), 500)
y_fit = logistic_function(x_fit, *params)

plt.scatter(x, y, color=colors[t])
plt.plot(x_fit, y_fit, color=colors[t], label=labels[t])
plt.legend()
plt.xlabel('Relative Error')
plt.ylabel('Action')
plt.title('Psychometric Curve')
plt.tight_layout()
plt.show()

# %%
f,ax = plt.subplots(2,1,figsize=(8,4))
before_states = np.array(store_states[0])
ax[0].plot(before_states[:,0], zorder=2, color='g', label='Model')
ax[0].scatter(np.arange(200), before_states[:,1], zorder=1, color='r', label='Ball')
ax[0].plot(np.arange(200),before_states[:,1], zorder=1, color='r', label='Ball',alpha=0.5)
ax[0].set_xlabel('Trial')
ax[0].set_ylabel('Location')
ax[0].set_title('Before learning Change-Point')
ax[0].legend()

after_states = np.array(store_states[-1])
ax[1].plot(after_states[:,0], zorder=2, color='g', label='Model')
ax[1].scatter(np.arange(200),after_states[:,1], zorder=1, color='r', label='Ball')
ax[1].plot(np.arange(200),after_states[:,1], zorder=1, color='r', label='Ball',alpha=0.5)
ax[1].set_xlabel('Trial')
ax[1].set_ylabel('Location')
ax[1].set_title('After learning Change-Point')
ax[1].legend()
f.tight_layout()
plt.show()
# %%
