# code to run model
# Ganesh, Adam

#%%
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from jax import grad, jit, vmap, random, lax, value_and_grad
from jax.nn import softmax, relu, tanh
from jax.nn.initializers import glorot_uniform, normal
from copy import deepcopy
from tasks import PIE_CP_OB


'''
To model the task in Nasser et al. 2021
0. Depends whether this is a change of point condition or odd ball task
1. At the beginning of each trial, the agent has time to move the bucket to a desired spot and press a confirmatory button to confirm location
2. helicopter drops the ball and trial ends
3. prediction error is shown for next trial

'''

# Define network training cycles
num_epochs = 100  # for training
test_epochs = []  # epochs during which neural network parameters are not updated but we let the network decide what to do.

# Define experiment params
num_context = 2 # context 0 is COP, context 1 is OB
num_trials = 100  # for each condition from Fig. 1
max_time = 300  # max time given to agent to move bucket and confirm the position before bag drop

# network params
obs_size = 4  # network input, what variables do we pass to the network as observations: bucket pos, bag pos, error
num_actions = 3 # move 0 - left, 1- right, 2- confirm location & begin ball drop

hidden_units = 128
gamma = 0.99  # play around with different gamma between 0.0 to 0.99
seed = 2024
eta =  0.0001

# Initialize model parameters
def initialize_params(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)

    n_input = obs_size+num_context

    Wxh =random.normal(k1, (n_input, hidden_units)) /jnp.sqrt(n_input)  # input to RNN weights
    Whh = random.normal(k2, (hidden_units, hidden_units)) /jnp.sqrt(hidden_units)  # RNN recurrent weights
    Wha = random.normal(k3, (hidden_units, num_actions)) /hidden_units  # RNN to actor output weights
    Whc = random.normal(k4, (hidden_units, 1)) /hidden_units # RNN to critic
    return [Wxh, Whh, Wha, Whc]

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

def compute_aprobs_and_values(params, state, prev_h):
    h = rnn_forward(params, state, prev_h)
    aprob, value = policy_and_value(params, h)
    return aprob, value

vmap_prob_val = vmap(compute_aprobs_and_values, in_axes=(None, 0, 0))

def td_loss(params, states, prev_hs, actions, rewards, gamma):
    aprobs, values = vmap_prob_val(params, jnp.array(states), jnp.array(prev_hs))
    log_likelihood = jnp.sum(jnp.log(aprobs)[:-1] * jnp.array(actions),axis=1)  # log probability of action as policy
    tde = jnp.array(compute_reward_prediction_error(jnp.array(rewards), jnp.array(values).reshape(-1), gamma))

    actor_loss = jnp.sum(log_likelihood * lax.stop_gradient(tde))  # maximize log policy * discounted reward
    critic_loss = -jnp.sum(tde ** 2) # minimize TD error
    tot_loss = actor_loss + 0.1 * critic_loss
    return tot_loss

@jit
def update_td_params(params, coords,prev_hs, actions, rewards, eta, gamma):
    loss, grads = value_and_grad(td_loss)(params, coords,prev_hs,actions, rewards, gamma)

    newparams = []
    for p,g in zip(params, grads):
        w = p + eta * g
        newparams.append(w)
    return newparams, grads, loss

def compute_reward_prediction_error(rewards, values, gamma=0.9):
    td = rewards + gamma * values[1:] - values[:-1]
    assert len(td.shape) == 1
    return td

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
def train(env, params, task_type, epoch):

    prev_h = random.normal(jax.random.PRNGKey(seed), (hidden_units,))*0.

    # to explicitly tell the network what context it is in.
    if task_type == "change-point":
        context =  np.array([1,0])
    elif task_type == "oddball":
        context =  np.array([0,1])


    perf = []
    store_states = []
    store_rnns = []
    store_actions = []
    store_values = []
    store_tde = []

    for trial in range(num_trials):
        
        obs, done = env.reset()
        norm_obs = env.normalize_states(obs) # to keep state to be between -1 to 1
        state = np.concatenate([norm_obs,context])

        states = []
        actions = []
        values = []
        rewards = []
        rnns = []
        prev_hs = []

        # give the agent time to move the bucket and press confirm, action == 2
        while not done:

            # pass states and contex to RNN
            h = rnn_forward(params, state, prev_h)

            # pass RNN activity to actor and critic to choose action
            policy, value = policy_and_value(params, h)
            action = get_onehot_action(policy)

            # pass action to environment
            next_obs, reward, done = env.step(np.argmax(action))

            # normalize 
            next_norm_obs = env.normalize_states(next_obs)
            next_state = np.concatenate([next_norm_obs,context])

            # print(trial, time, action, state, reward, done)

            # store states, action, reward for training
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            rnns.append(h)
            values.append(value)
            prev_hs.append(prev_h)

            # make sure you assign the state and rnn state correctly for the next trial
            state = next_state.copy()
            prev_h = h.copy()

        # if epoch == 0 or epoch == num_epochs-1: 
        #     print(f"Trial: {trial}, Time to confirm: {time}, Last obs: {next_obs}, Reward: {reward:.3f}")

        #update params after each trial
        states.append(next_state) # predict value of next state in TD update.
        prev_hs.append(h)

        # print(np.array(states).shape, np.array(prev_hs).shape, np.array(actions).shape, np.array(rewards).shape, np.array(values).shape)

        if epoch not in test_epochs:
            params, grads, loss = update_td_params(params, states, prev_hs, actions, rewards, eta, gamma)

        perf.append(np.array([reward, env.time, loss]))
        store_states.append(np.array(states)[:-1])
        store_rnns.append(np.array(rnns))
        store_actions.append(np.argmax(np.array(actions),axis=0))
        # store_values.append(np.array(values))
        # store_tde.append(np.array(compute_reward_prediction_error(rewards, np.array(values).reshape(-1), gamma)))  # reward prediction error, similar to Dopamine activity

    if epoch == 0 or epoch == num_epochs-1:
        env.render()

    return params, perf, store_states, store_rnns, store_actions, store_values, store_tde

# Helicopter task
# Initialize parameters & optimizer
params = initialize_params(jax.random.PRNGKey(seed))
initparams = deepcopy(params)

all_perfs = []

# Train the model
for epoch in range(num_epochs):

    perfs = []

    for task_type in ["change-point", "oddball"]:

        env = PIE_CP_OB(condition=task_type,  total_trials=num_trials,max_time=max_time, train_cond=True)

        params, perf, store_states, store_rnns, store_actions, store_values, store_tde = train(env, params, task_type, epoch)

        print(f'### Epoch: {epoch}, Task: {task_type.capitalize()}, R: {np.round(np.mean(perf,axis=0))}')
        perfs.append(perf)

    all_perfs.append(np.mean(np.array(perfs),axis=1))


f,axs = plt.subplots(3,1,figsize=(3,2*3))
ylabels = ['Reward','Time to confirm','Loss']  # increase, increase, decrease
for i in range(3):
    axs[i].plot(np.mean(np.array(all_perfs),axis=1)[:,i])
    axs[i].set_xlabel('Epoch')
    axs[i].set_ylabel(ylabels[i])
plt.tight_layout()