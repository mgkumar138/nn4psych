# code to run model
# Ganesh, Adam
#two arm bandit no context given

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
num_trials = 100
num_contexts = 3
num_actions = 2
hidden_units = 64
gamma = 0.95

# Define reward probabilities for each context and each arm
reward_probs = jnp.array([
    [0.9, 0.1],  # Context 1
    [0.2, 0.8],  # Context 2
    [0.6, 0.4],  # Context 3
])

# Initialize model parameters
def initialize_params(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    Wxh =random.normal(k1, (3, hidden_units)) /jnp.sqrt(hidden_units)
    Whh = random.normal(k2, (hidden_units, hidden_units)) /jnp.sqrt(hidden_units)
    Wha = random.normal(k3, (hidden_units, num_actions)) *1e-3
    Whc = random.normal(k4, (hidden_units, 1)) * 1e-3
    return Wxh, Whh, Wha, Whc

params = initialize_params(jax.random.PRNGKey(0))
initparams = deepcopy(params)

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


# Optimizer
optimizer = optax.adam(1e-2)

# Training loop
def train(params, context, reward_prob):
    
    prev_h = random.normal(jax.random.PRNGKey(0), (hidden_units,))*0.1
    opt_state = optimizer.init(params)

    loss_history = []
    reward_history = []
    state = np.zeros_like(context)
    weights = []
    actions = []
    for trial in range(num_trials):

        h = rnn_forward(params, state, prev_h)
        policy, _ = policy_and_value(params, h)
        action = get_onehot_action(policy)

        # pass action to env to get next state and reward 
        rprob = reward_prob[np.argmax(action)]
        reward = np.random.choice([0, 1], p=np_softmax([1 - rprob, rprob]))
        next_state = np.zeros_like(context) # no input to the RNN as of now

        # get next state value prediction
        new_h = rnn_forward(params, next_state, h)
        _, next_value = policy_and_value(params, new_h)

        # compute the loss with respect to the state, action, reward and newstate
        loss, grads = jax.value_and_grad(loss_fn)(params, state, next_value, prev_h, action, reward)
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # make sure you assign the state and rnn state correctly for the next trial
        state = next_state
        prev_h = h

        loss_history.append(loss)
        reward_history.append(reward)

        print(trial, np.round(policy,1), reward,loss, grads[1][0,0])

        weights.append(h)
        actions.append(action)

    return params, weights, actions, loss_history, reward_history

#%%
# Generate synthetic data (context, action, reward tuples)
np.random.seed(0)
context = np.eye(num_contexts)[0]
reward_prob = reward_probs[0]

# Train the model
params, weights, actions, loss_history, reward_history = train(params, context, reward_prob)

#save weights np.array(weights).T
w_array = np.array(weights).T
np.save('weights.npy', w_array)

action_array = np.array(actions).T
np.save('actions.npy', action_array)

#%%
# Plot the reward over trials
f,ax = plt.subplots(2,2)
ax[0,0].plot(moving_average(reward_history, window_size=int(num_trials*0.1)), label='MA Reward')
ax[0,0].set_xlabel('Trial')
ax[0,0].set_ylabel('Reward')
ax[0,0].set_title('Reward over Trials')

ax[0,1].plot(loss_history, label='TD error', color='tab:orange')
ax[0,1].set_xlabel('Trial')
ax[0,1].set_ylabel('Loss')
ax[0,1].set_title('Actor-Critic Loss over Trials')


im = ax[1,0].imshow(params[2].T,aspect='auto')
plt.colorbar(im,ax=ax[1,0])
ax[1,0].set_xlabel('Action')
ax[1,0].set_ylabel('Hidden units')

ax[1,1].plot(params[3])
ax[1,1].set_xlabel('Hidden units')
ax[1,1].set_ylabel('Value')
f.tight_layout()
# %%
