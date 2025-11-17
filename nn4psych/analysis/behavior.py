"""
Behavior extraction and analysis utilities.
"""

from typing import Tuple, List, Dict, Optional
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from nn4psych.models.actor_critic import ActorCritic
from nn4psych.envs.predictive_inference import PIE_CP_OB_v2
from nn4psych.utils.metrics import get_lrs_v2


def extract_behavior(
    model: ActorCritic,
    env: PIE_CP_OB_v2,
    n_epochs: int = 100,
    n_trials: int = 200,
    reset_memory: bool = True,
    preset_memory: float = 0.0,
    device: Optional[torch.device] = None,
) -> Tuple:
    """
    Extract behavioral data by running model through task.

    Parameters
    ----------
    model : ActorCritic
        Trained model to evaluate.
    env : PIE_CP_OB_v2
        Task environment.
    n_epochs : int, optional
        Number of epochs to run. Default is 100.
    n_trials : int, optional
        Number of trials per epoch. Default is 200.
    reset_memory : bool, optional
        Whether to reset hidden state between epochs. Default is True.
    preset_memory : float, optional
        Value to preset hidden state to. Default is 0.0.
    device : torch.device, optional
        Device to run model on.

    Returns
    -------
    all_states : list
        List of state tuples for each epoch.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_states = []

    with torch.no_grad():
        for epoch in range(n_epochs):
            # Reset hidden state
            if reset_memory:
                h = model.reset_hidden(batch_size=1, device=device, preset_value=preset_memory)
            else:
                if epoch == 0:
                    h = model.get_initial_hidden(batch_size=1, device=device)

            # Reset environment for epoch
            env._reset_state()

            # Run trials
            for trial in range(n_trials):
                obs, done = env.reset()
                norm_obs = env.normalize_states(obs)
                state = np.concatenate([norm_obs, env.context])

                while not done:
                    # Get model action
                    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    actor_logits, _, h = model(x, h)
                    action_probs = torch.softmax(actor_logits, dim=-1)
                    action = torch.argmax(action_probs, dim=-1).item()

                    # Take action
                    obs, reward, done = env.step(action)
                    norm_obs = env.normalize_states(obs)
                    state = np.concatenate([norm_obs, env.context])

            # Store epoch data
            states = env.get_state_history()
            all_states.append(states)

    return all_states


def get_area(
    model_path: Path,
    epochs: int = 100,
    reset_memory: bool = True,
    model_class: type = ActorCritic,
    input_dim: int = 9,
    hidden_dim: int = 64,
    action_dim: int = 3,
    env_params: Optional[Dict] = None,
) -> float:
    """
    Calculate performance area under learning curve for a trained model.

    Parameters
    ----------
    model_path : Path
        Path to model weights file.
    epochs : int, optional
        Number of evaluation epochs.
    reset_memory : bool, optional
        Whether to reset memory between epochs.
    model_class : type, optional
        Model class to instantiate.
    input_dim : int, optional
        Model input dimension.
    hidden_dim : int, optional
        Model hidden dimension.
    action_dim : int, optional
        Model action dimension.
    env_params : dict, optional
        Parameters for environment initialization.

    Returns
    -------
    float
        Performance metric (mean learning rate area).
    """
    # Load model
    model = model_class(input_dim, hidden_dim, action_dim)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    # Default environment parameters
    if env_params is None:
        env_params = {
            'total_trials': 200,
            'max_time': 300,
            'train_cond': False,
            'max_displacement': 10,
            'reward_size': 5.0,
        }

    # Create environments
    env_cp = PIE_CP_OB_v2(condition='change-point', **env_params)
    env_ob = PIE_CP_OB_v2(condition='oddball', **env_params)

    # Extract behavior
    states_cp = extract_behavior(
        model, env_cp, n_epochs=epochs, reset_memory=reset_memory
    )
    states_ob = extract_behavior(
        model, env_ob, n_epochs=epochs, reset_memory=reset_memory
    )

    # Calculate learning rates
    lr_areas = []
    for epoch in range(epochs):
        for states in [states_cp[epoch], states_ob[epoch]]:
            pes, lrs = get_lrs_v2(states, threshold=20)
            valid_lrs = lrs[lrs >= 0]
            if len(valid_lrs) > 0:
                lr_areas.append(np.mean(valid_lrs))

    return np.mean(lr_areas) if lr_areas else 0.0


def batch_extract_behavior(
    model_paths: List[Path],
    n_epochs: int = 100,
    reset_memory: bool = True,
    show_progress: bool = True,
) -> Dict[str, List]:
    """
    Extract behavior from multiple models.

    Parameters
    ----------
    model_paths : list of Path
        List of paths to model weight files.
    n_epochs : int, optional
        Number of epochs per model.
    reset_memory : bool, optional
        Whether to reset memory between epochs.
    show_progress : bool, optional
        Whether to show progress bar.

    Returns
    -------
    dict
        Dictionary with 'cp' and 'ob' keys containing state lists.
    """
    results = {'cp': [], 'ob': [], 'models': []}

    iterator = tqdm(model_paths, desc="Extracting behavior") if show_progress else model_paths

    for model_path in iterator:
        # Load model
        model = ActorCritic(9, 64, 3)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)

        # Extract for both conditions
        env_cp = PIE_CP_OB_v2(condition='change-point')
        env_ob = PIE_CP_OB_v2(condition='oddball')

        states_cp = extract_behavior(model, env_cp, n_epochs=n_epochs, reset_memory=reset_memory)
        states_ob = extract_behavior(model, env_ob, n_epochs=n_epochs, reset_memory=reset_memory)

        results['cp'].append(states_cp)
        results['ob'].append(states_ob)
        results['models'].append(str(model_path))

    return results
