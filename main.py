# code to run model
# Ganesh, Adam

#%%
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import grad, jit, vmap
from jax.nn import softmax, relu
from jax.nn.initializers import glorot_uniform, normal

#%%
# Define constants
num_trials = 120
num_contexts = 3
num_actions = 2
hidden_units = 64
gamma = 0.99

# Define reward probabilities for each context and each arm
reward_probs = jnp.array([
    [0.9, 0.1],  # Context 1
    [0.2, 0.8],  # Context 2
    [0.6, 0.4],  # Context 3
])

# Initialize model parameters
def initialize_params(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    Wxh = glorot_uniform()(k1, (3, hidden_units))
    Whh = glorot_uniform()(k2, (hidden_units, hidden_units))
    Wha = glorot_uniform()(k3, (hidden_units, num_actions))
    Whc = glorot_uniform()(k4, (hidden_units, 1))
    return Wxh, Whh, Wha, Whc

params = initialize_params(jax.random.PRNGKey(0))

# Recurrent Neural Network forward pass
def rnn_forward(params, inputs, h):
    Wxh, Whh, Wha, Whc = params
    h = relu(jnp.dot(inputs, Wxh) + jnp.dot(h, Whh))
    return h

# Define policy (actor) and value (critic) functions
def policy_and_value(params, h):
    Wxh, Whh, Wha, Whc = params
    policy = jnp.dot(h, Wha)
    value = jnp.dot(h, Whc) # Critic
    policy_prob = softmax(policy)  # Actor
    return policy_prob, value, h

def sample_action(policy_prob):


# Loss function
def loss_fn(params, contexts, action_prob, actions, rewards, values, next_values):
    td_errors = rewards + gamma * next_values - values
    policy_losses = -jnp.log(action_prob) * actions * td_errors
    value_losses = td_errors ** 2
    loss = jnp.mean(policy_losses) + jnp.mean(value_losses)
    return loss

# Optimizer
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# Training loop
def train(params, opt_state, contexts, actions, rewards):
    h = jnp.zeros(hidden_units)
    loss_history = []

    for trial in range(num_trials):
        inputs = contexts[trial]
        reward = rewards[trial]
        action = actions[trial]

        policy, value, h = policy_and_value(params, inputs, h)
        next_policy, next_value, _ = policy_and_value(params, inputs, h)
        
        action_prob = policy[action]
        loss, grads = jax.value_and_grad(loss_fn)(params, inputs, action_prob, reward, value, next_value)
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        loss_history.append(loss)
    return params, loss_history

# Generate synthetic data (context, action, reward tuples)
np.random.seed(0)
contexts = np.eye(num_contexts)[np.random.choice(num_contexts, num_trials)]
actions = np.random.randint(2, size=num_trials)  # Random actions (0 or 1)
rewards = [reward_probs[c, a] for c, a in zip(np.argmax(contexts, axis=1), actions)]

# Train the model
params, loss_history = train(params, opt_state, contexts, actions, rewards)

# Plot the reward over trials
plt.plot(loss_history)
plt.xlabel('Trial')
plt.ylabel('TD Error Loss')
plt.title('TD Error Loss over Trials')
plt.show()