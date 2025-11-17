"""
Metrics and state extraction utilities for behavior analysis.
"""

from typing import Tuple
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy import stats


def get_lrs(states: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and smooth learning rates from state data.

    Parameters
    ----------
    states : tuple
        State tuple containing:
        - [0]: trials
        - [1]: bucket_positions (predicted state)
        - [2]: bag_positions (true state)
        - [3]: helicopter_positions
        - [4]: hazard_triggers

    Returns
    -------
    prediction_error_sorted : np.ndarray
        Sorted absolute prediction errors.
    smoothed_learning_rate : np.ndarray
        Smoothed learning rates corresponding to sorted errors.

    Notes
    -----
    Learning rate is computed as: update / |prediction_error|
    where update = |diff(predicted_state)|.
    """
    true_state = states[2]  # bag position
    predicted_state = states[1]  # bucket position

    prediction_error = np.abs(true_state - predicted_state)[:-1]
    update = np.abs(np.diff(predicted_state))
    learning_rate = np.where(prediction_error != 0, update / prediction_error, 0)

    # Sort by prediction error
    sorted_indices = np.argsort(prediction_error)
    prediction_error_sorted = prediction_error[sorted_indices]
    learning_rate_sorted = learning_rate[sorted_indices]

    # Smooth learning rates
    window_size = 10
    smoothed_learning_rate = uniform_filter1d(learning_rate_sorted, size=window_size)

    return prediction_error_sorted, smoothed_learning_rate


def get_lrs_v2(
    states: Tuple,
    threshold: float = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract learning rates with threshold filtering and padding.

    This version filters out small prediction errors and clips learning rates
    to [0, 1] range.

    Parameters
    ----------
    states : tuple
        State tuple (same as get_lrs).
    threshold : float, optional
        Minimum absolute prediction error to include. Default is 20.

    Returns
    -------
    pad_pes : np.ndarray
        Padded prediction errors (-1 for padding).
    pad_lrs : np.ndarray
        Padded learning rates (-1 for padding).

    Notes
    -----
    This version:
    1. Uses signed prediction errors for learning rate calculation
    2. Filters out zero prediction errors
    3. Clips learning rates to [0, 1]
    4. Only includes errors above threshold
    5. Pads arrays to original length with -1
    """
    true_state = states[2]  # bag position
    predicted_state = states[1]  # bucket position

    # Calculate signed prediction error and update
    prediction_error = (true_state - predicted_state)[:-1]
    update = np.diff(predicted_state)

    # Remove zero prediction errors
    idx = prediction_error != 0
    prediction_error = prediction_error[idx]
    update = update[idx]

    # Calculate learning rate (can be negative)
    learning_rate = update / prediction_error

    # Filter by absolute prediction error threshold
    abs_pe = np.abs(prediction_error)
    idx = abs_pe > threshold
    pes = abs_pe[idx]
    lrs = np.clip(learning_rate, 0, 1)[idx]

    # Sort by prediction error
    sorted_indices = np.argsort(pes)
    prediction_error_sorted = pes[sorted_indices]
    learning_rate_sorted = lrs[sorted_indices]

    # Pad to original length
    pad_length = len(true_state) - len(prediction_error_sorted) - 1
    pad_pes = np.pad(
        prediction_error_sorted,
        (0, pad_length),
        'constant',
        constant_values=-1,
    )
    pad_lrs = np.pad(
        learning_rate_sorted,
        (0, pad_length),
        'constant',
        constant_values=-1,
    )

    return pad_pes, pad_lrs


def extract_states(states: Tuple) -> Tuple:
    """
    Extract prediction error, learning rate, and hazard information from states.

    Parameters
    ----------
    states : tuple
        State tuple containing trial data.

    Returns
    -------
    prediction_error : np.ndarray
        Absolute prediction errors (capped at 100).
    update : np.ndarray
        Absolute updates in predicted state.
    learning_rate : np.ndarray
        Learning rates (update / prediction_error).
    true_state : np.ndarray
        True bag positions.
    predicted_state : np.ndarray
        Predicted bucket positions.
    hazard_distance : np.ndarray
        Distance from last hazard event.
    hazard_trials : np.ndarray
        Binary array indicating hazard events.

    Warnings
    --------
    This function is partially deprecated. Consider using get_lrs_v2 for
    learning rate extraction.
    """
    true_state = states[2]  # bag position
    predicted_state = states[1]  # bucket position

    prediction_error = np.abs(true_state - predicted_state)
    prediction_error = np.minimum(prediction_error, 100)
    prediction_error = prediction_error[:-1]

    update = np.abs(np.diff(predicted_state))
    learning_rate = np.where(prediction_error != 0, update / prediction_error, 0)

    hazard_trials = states[4]
    hazard_indexes = np.where(states[4] == 1)[0]
    hazard_distance = np.zeros(len(states[0]), dtype=int)

    current = 0
    for i in range(len(states[0])):
        if i in hazard_indexes:
            current = 0
        hazard_distance[i] = current
        current += 1

    return (
        prediction_error,
        update,
        learning_rate,
        true_state,
        predicted_state,
        hazard_distance,
        hazard_trials,
    )


def calculate_alpha_changepoint(omega: float, tau: float) -> float:
    """
    Calculate learning rate for changepoint model.

    Equation: alpha = omega + tau - (omega * tau)

    Parameters
    ----------
    omega : float
        Changepoint probability.
    tau : float
        Relative uncertainty.

    Returns
    -------
    float
        Learning rate alpha.
    """
    return omega + tau - (omega * tau)


def calculate_alpha_oddball(tau: float, omega: float) -> float:
    """
    Calculate learning rate for oddball model.

    Equation: alpha = tau - (tau * omega)

    Parameters
    ----------
    tau : float
        Relative uncertainty.
    omega : float
        Changepoint probability.

    Returns
    -------
    float
        Learning rate alpha.
    """
    return tau - (tau * omega)


def calculate_omega(H: float, U_val: float, N_val: float) -> float:
    """
    Calculate updated changepoint probability (omega).

    Equation: omega = (H * U_val) / (H * U_val + (1 - H) * N_val)

    Parameters
    ----------
    H : float
        Prior hazard rate.
    U_val : float
        Uniform PDF value.
    N_val : float
        Normal PDF value.

    Returns
    -------
    float
        Updated omega value.
    """
    numerator = H * U_val
    denominator = H * U_val + (1 - H) * N_val
    return numerator / denominator


def calculate_tau(tau: float, UU: float) -> float:
    """
    Update relative uncertainty (tau).

    Equation: tau = tau / UU

    Parameters
    ----------
    tau : float
        Current relative uncertainty.
    UU : float
        Uncertainty underestimation factor.

    Returns
    -------
    float
        Updated tau value.
    """
    return tau / UU


def calculate_normative_update(alpha: float, delta: float) -> float:
    """
    Calculate normative update.

    Equation: normative_update = alpha * delta

    Parameters
    ----------
    alpha : float
        Learning rate.
    delta : float
        Prediction error.

    Returns
    -------
    float
        Normative update value.
    """
    return alpha * delta


def calculate_sigma_update(
    sigma_motor: float,
    normative_update: float,
    sigma_LR: float,
) -> float:
    """
    Calculate variability of update.

    Equation: sigma_update = sigma_motor + normative_update * sigma_LR

    Parameters
    ----------
    sigma_motor : float
        Motor noise.
    normative_update : float
        Normative update value.
    sigma_LR : float
        Learning rate noise.

    Returns
    -------
    float
        Updated sigma value.
    """
    return sigma_motor + normative_update * sigma_LR


def calculate_L_normative(
    participant_update: np.ndarray,
    normative_update: float,
    sigma_update: float,
) -> float:
    """
    Calculate normative likelihood.

    Parameters
    ----------
    participant_update : np.ndarray
        Participant's update data.
    normative_update : float
        Normative update value.
    sigma_update : float
        Updated sigma value.

    Returns
    -------
    float
        Log-normalized likelihood.
    """
    return stats.norm.pdf(
        participant_update,
        loc=normative_update,
        scale=sigma_update,
    )
