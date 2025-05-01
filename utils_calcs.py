import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.nn import init
from tasks import PIE_CP_OB_v2

#bayesian_model specific equations (move back into model_bayesian.py)
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

#lr in analyze_hyperparams (standardize to the rest)
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

#lr elsewhere
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

def get_lrs_v3(states, threshold=0):
    '''
    -takes in state vector
    -threshold is the cutoff for prediction error to be considered a learning rate
    -returns prediction error and learning rate sorted by prediction error
    '''
    true_state = states[2]  # bag position
    predicted_state = states[1]  # bucket position
    prediction_error = abs((true_state - predicted_state)[:-1])
    update = np.diff(predicted_state)

    #index 1 - nonzero division check
    # idx = prediction_error != 0
    # prediction_error = prediction_error[idx]
    # update = update[idx]
    # learning_rate = abs(update / prediction_error)
    #option 2 - just clip the prediction error to avoid division by zero
    prediction_error = np.clip(prediction_error, 1, None)
    learning_rate = abs(update / prediction_error)

    #index 2- pe threshold
    idx = prediction_error >= threshold
    pes = prediction_error[idx]
    lrs = np.clip(learning_rate, 0, 1)[idx]
    #sort for easy plotting
    sorted_indices = np.argsort(pes)
    prediction_error_sorted = pes[sorted_indices]
    learning_rate_sorted = lrs[sorted_indices]

    area = np.trapz(learning_rate_sorted, prediction_error_sorted)

    return prediction_error_sorted, learning_rate_sorted, pes, lrs, area

#small calcs
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

def compute_update_ratios(lrs):
    """
    Compute the proportion of non-updates and moderate updates.
    Non-update: lr < 0.1, Moderate update: 0.1 <= lr < 0.9.
    """
    if len(lrs) == 0:
        return 0, 0
    p_non = np.mean(lrs < 0.1)
    p_med = np.mean((lrs >= 0.1) & (lrs < 0.9))
    p_total = np.mean(lrs >= 0.9)
    return p_med, p_non, p_total






