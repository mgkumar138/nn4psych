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
from tasks import DiscretePredictiveInferenceEnv



# Define constants
num_trials = 100 # per trial
num_actions = 2
hidden_units = 64
gamma = 0.0
learning_rate = 1e-4
seed = 2024

reward_feedback = True
action_feedback = True
context_feedback = True


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
def train(params, prev_h, task_type, history):

    env = DiscretePredictiveInferenceEnv(condition=task_type) #DiscretePredictiveInferenceEnv(condition=task_type)
        
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        
        h = rnn_forward(params, state, prev_h)
        policy, _ = policy_and_value(params, h)
        action = get_onehot_action(policy)

        # pass action to env to get next state and reward 
        rprob = reward_prob[np.argmax(action)]
        reward = np.random.choice([0, 1], p=[1 - rprob, rprob])

        # update state
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward


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

        history.append([reward, np.argmax(action), loss])


    return params, history

# contextual bandit training
# Initialize parameters & optimizer
params = initialize_params(jax.random.PRNGKey(seed))
initparams = deepcopy(params)
optimizer = optax.adam(learning_rate)

history = []

# Train the model
for epoch in range(num_epochs):
    prev_h = random.normal(jax.random.PRNGKey(0), (hidden_units,))*0.1
    opt_state = optimizer.init(params)

    for context in range(num_contexts):
        # depending on the context, determine the reward probabilities
        print(f'### Epoch {epoch} Context {context}')
        reward_prob = reward_probs[context]
        params, history = train(params, context, reward_prob, opt_state, prev_h, history)

