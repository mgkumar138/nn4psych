import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.nn import init
from tasks import PIE_CP_OB_v2


def calculate_normative_update(alpha, delta):
    """
    Equation 1: Calculate normative update.
    
    Parameters:
        alpha (float): Learning rate.
        delta (float): Prediction error.
        t (int): Current time step.
    
    Returns:
        float: Normative update value.
    
    Equation:
        normative_update[t] = alpha[t] * delta[t]
    """
    return alpha * delta

def calculate_alpha_changepoint(omega, tau):
    """
    Equation 2: Calculate alpha for changepoint model.
    
    Parameters:
        omega (float): Changepoint probability.
        tau (float): Relative uncertainty.
        t (int): Current time step.
    
    Returns:
        float: Updated alpha value.
    
    Equation:
        alpha[t] = omega + tau - (omega * tau)
    """
    return omega + tau - (omega * tau)

def calculate_alpha_oddball(tau, omega):
    """
    Equation 3: Calculate alpha for oddball model.
    
    Parameters:
        tau (float): Relative uncertainty.
        omega (float): Changepoint probability.
    
    Returns:
        float: Updated alpha value.
    
    Equation:
        alpha[t] = tau - (tau * omega)
    """
    return tau - (tau * omega)

def calculate_omega(H, U_val, N_val):
    """
    Equation 4: Calculate updated omega.
    
    Parameters:
        H (float): Probability depending on the condition.
        U_val (float): Uniform PDF value raised to the likelihood weight.
        N_val (float): Normal PDF value raised to the likelihood weight.
    
    Returns:
        float: Updated omega value.
    
    Equation:
        omega = (H * U_val) / (H * U_val + (1 - H) * N_val)
    """
    return (H * U_val) / (H * U_val + (1 - H) * N_val)

def calculate_tau(tau, UU):
    """
    Equation 5: Update tau based on uncertainty underestimation.
    
    Parameters:
        tau (float): Relative uncertainty.
        UU (float): Uncertainty underestimation.
    
    Returns:
        float: Updated tau value.
    
    Equation:
        tau = tau / UU
    """
    return tau / UU

def calculate_L_normative(participant_update, normative_update, sigma_update):
    """
    Equation 6: Calculate normative likelihood.
    
    Parameters:
        participant_update (numpy.ndarray): Participant's update data.
        normative_update (float): Normative update value.
        sigma_update (float): Updated sigma value.
        t (int): Current time step.
    
    Returns:
        float: Log-normalized likelihood.
    
    Equation:
        L_normative = stats.norm.pdf(participant_update[t], loc=normative_update[t], scale=sigma_update)
    """
    return stats.norm.pdf(participant_update, loc=normative_update, scale=sigma_update)

def calculate_sigma_update(sigma_motor, normative_update, sigma_LR):
    """
    Equation 7: Calculate variability of update.
    
    Parameters:
        sigma_motor (float): Motor sigma value.
        normative_update (float): Normative update value.
        sigma_LR (float): Learning rate sigma value.
        t (int): Current time step.
    
    Returns:
        float: Updated sigma value.
    
    Equation:
        sigma_update = sigma_motor + normative_update[t] * sigma_LR
    """
    return sigma_motor + normative_update * sigma_LR

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, gain=1.5, noise=0.0, bias=False):
        super(ActorCritic, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gain = gain
        self.noise = noise  # Include the noise variance as an argument
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity='tanh',bias=bias)
        self.actor = nn.Linear(hidden_dim, action_dim,bias=bias)
        self.critic = nn.Linear(hidden_dim, 1,bias=bias)
        self.init_weights()

    def init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                init.normal_(param, mean=0, std=1/(self.input_dim**0.5))
            elif 'weight_hh' in name:
                init.normal_(param, mean=0, std=self.gain / self.hidden_dim**0.5)
            elif 'bias_ih' in name or 'bias_hh' in name:
                init.constant_(param, 0)

        for layer in [self.actor, self.critic]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.normal_(param, mean=0, std=1/self.hidden_dim)
                elif 'bias' in name:
                    init.constant_(param, 0)

    def forward(self, x, hx):
        r, h = self.rnn(x, hx)
        r = r.squeeze(1)
        critic_value = self.critic(r)

        return self.actor(r), critic_value, h



def get_lrs_analyze_hyperparams(states):
    epochs = states.shape[0]
    pess, lrss, area = [],[], []
    for c in range(2):
        pes,lrs = [],[]
        for e in range(epochs):
            pe, lr = get_lrs_v2_analyze_hyperparams(states[e, c])

            pes.append(pe)
            lrs.append(lr)

        pes = np.concatenate(pes)
        lrs = np.concatenate(lrs)
        sorted_indices = np.argsort(pes)
        prediction_error_sorted = pes[sorted_indices]
        learning_rate_sorted = lrs[sorted_indices]

        pess.append(prediction_error_sorted)
        lrss.append(learning_rate_sorted)
        area.append(np.trapz(learning_rate_sorted, prediction_error_sorted))
    return area, pess, lrss



def get_lrs_v2_analyze_hyperparams(states, threshold=20):
    '''
    taken from analyze_hyperparams, needs cleanup
    different from other get_lrs_v2
    '''
    true_state = states[2]  # bag position
    predicted_state = states[1]  # bucket position
    prediction_error = (true_state - predicted_state)[:-1]
    update = np.diff(predicted_state)

    idx = prediction_error !=0
    prediction_error= prediction_error[idx]
    update = update[idx]
    learning_rate = update / prediction_error

    prediction_error = abs(prediction_error)
    idx = prediction_error>threshold
    pes = prediction_error[idx]
    lrs = np.clip(learning_rate,0,1)[idx]

    sorted_indices = np.argsort(pes)
    prediction_error_sorted = pes[sorted_indices]
    learning_rate_sorted = lrs[sorted_indices]

    return prediction_error_sorted, learning_rate_sorted







def get_lrs(states):
    true_state = states[2]  # bag position
    predicted_state = states[1]  # bucket position
    prediction_error = abs((true_state - predicted_state))[:-1]
    update = abs(np.diff(predicted_state))
    learning_rate = np.where(prediction_error !=0, update / prediction_error)
    
    sorted_indices = np.argsort(prediction_error)
    prediction_error_sorted = prediction_error[sorted_indices]
    learning_rate_sorted = learning_rate[sorted_indices]

    window_size = 10
    smoothed_learning_rate = uniform_filter1d(learning_rate_sorted, size=window_size)
    return prediction_error_sorted, smoothed_learning_rate

def get_lrs_v2(states, threshold=20):
    true_state = states[2]  # bag position
    predicted_state = states[1]  # bucket position
    prediction_error = (true_state - predicted_state)[:-1]
    update = np.diff(predicted_state)

    idx = prediction_error !=0
    prediction_error= prediction_error[idx]
    update = update[idx]
    learning_rate = update / prediction_error

    prediction_error = abs(prediction_error)
    idx = prediction_error>threshold
    pes = prediction_error[idx]
    lrs = np.clip(learning_rate,0,1)[idx]

    sorted_indices = np.argsort(pes)
    prediction_error_sorted = pes[sorted_indices]
    learning_rate_sorted = lrs[sorted_indices]

    pad_pes = np.pad(prediction_error_sorted,(0, len(true_state)-len(prediction_error_sorted)-1), 'constant', constant_values=-1)
    pad_lrs = np.pad(learning_rate_sorted,(0, len(true_state)-len(learning_rate_sorted)-1), 'constant', constant_values=-1)

    return pad_pes, pad_lrs

def get_mean_ci(x, valididx):
    m = []
    s = []
    numparams = x.shape[0]
    for p in range(numparams):
        idx = int(valididx[p])
        m.append(np.mean(x[p,:idx],axis=0))
        s.append(np.std(x[p,:idx],axis=0)/np.sqrt(idx))
    m = np.array(m)
    s = np.array(s)
    return m, s


def plot_behavior(states, context,epoch, ax=None):
    if ax is None:
        plt.figure(figsize=(10, 6))
    trials, bucket_positions, bag_positions, helicopter_positions, hazard_triggers = states
    # plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='blue')
    plt.plot(trials, bag_positions, label='Bag', color='red', marker='o', linestyle='-.', alpha=0.5, ms=2)
    plt.plot(trials, helicopter_positions, label='Heli', color='green', linestyle='--',ms=2)
    plt.plot(trials, bucket_positions, label='Bucket', color='b',marker='o', linestyle='-.', alpha=0.5,ms=2)

    plt.ylim(-10, 310)  # Set y-axis limit from 0 to 300
    plt.xlabel('Trial')
    plt.ylabel('Position')
    plt.title(f"{context}, E:{epoch}")
    plt.legend(fontsize=6)

def get_area(model_path, epochs=100, reset_memory=0.0):
    hidden_dim = 64
    trials = 200

    model = ActorCritic(9, hidden_dim, 3)
    model.load_state_dict(torch.load(model_path))

    contexts = ["change-point", "oddball"]

    all_states = np.zeros([epochs, 2, 5, trials])
    for epoch in range(epochs):
        for tt, context in enumerate(contexts):
            env = PIE_CP_OB_v2(condition=context, max_time=300, total_trials=trials, 
                               train_cond=False, max_displacement=10, reward_size=2)

            hx = torch.randn(1, 1, hidden_dim) * 1 / hidden_dim**0.5
            for trial in range(trials):

                next_obs, done = env.reset()
                norm_next_obs = env.normalize_states(next_obs)
                next_state = np.concatenate([norm_next_obs, env.context, np.array([0.0])])
                next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

                hx = hx.detach()

                while not done:

                    if np.random.random_sample() < reset_memory:
                        hx = (torch.randn(1, 1, hidden_dim) * 1 / hidden_dim**0.5)

                    actor_logits, critic_value, hx = model(next_state, hx)
                    probs = Categorical(logits=actor_logits)
                    action = probs.sample()

                    next_obs, reward, done = env.step(action.item())

                    norm_next_obs = env.normalize_states(next_obs)
                    next_state = np.concatenate([norm_next_obs, env.context, np.array([reward])])
                    next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

            all_states[epoch, tt] = np.array([env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])

    return all_states


