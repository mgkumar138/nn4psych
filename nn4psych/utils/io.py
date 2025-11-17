"""
Input/Output utilities for saving and loading models and data.
"""

import pickle
from pathlib import Path
from typing import Any, Union, Optional

import torch


def saveload(
    filename: Union[str, Path],
    variable: Any = None,
    opt: str = "load",
) -> Optional[Any]:
    """
    Save or load a variable using pickle.

    Parameters
    ----------
    filename : str or Path
        Path to the pickle file (without .pickle extension).
    variable : Any, optional
        Variable to save (required if opt='save').
    opt : str, optional
        Operation type: 'save' or 'load'. Default is 'load'.

    Returns
    -------
    Any or None
        Loaded variable if opt='load', None if opt='save'.

    Examples
    --------
    >>> data = {'key': 'value', 'numbers': [1, 2, 3]}
    >>> saveload('my_data', data, 'save')
    >>> loaded_data = saveload('my_data', opt='load')
    """
    filepath = Path(filename)
    if not filepath.suffix:
        filepath = filepath.with_suffix('.pickle')

    if opt == 'save':
        with open(filepath, "wb") as file:
            pickle.dump(variable, file)
        print(f'File saved: {filepath}')
        return None
    else:  # load
        with open(filepath, "rb") as file:
            return pickle.load(file)


def save_model(
    model: torch.nn.Module,
    filepath: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save PyTorch model weights and optional metadata.

    Parameters
    ----------
    model : torch.nn.Module
        Model to save.
    filepath : str or Path
        Path to save the model.
    optimizer : torch.optim.Optimizer, optional
        Optimizer state to save.
    metadata : dict, optional
        Additional metadata to save with the model.

    Examples
    --------
    >>> from nn4psych.models import ActorCritic
    >>> model = ActorCritic(9, 64, 3)
    >>> save_model(model, 'model_weights.pth', metadata={'epochs': 100})
    """
    filepath = Path(filepath)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': model.input_dim,
            'hidden_dim': model.hidden_dim,
            'action_dim': model.action_dim,
            'gain': model.gain,
            'noise': model.noise,
        }
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if metadata is not None:
        checkpoint['metadata'] = metadata

    torch.save(checkpoint, filepath)
    print(f'Model saved: {filepath}')


def load_model(
    filepath: Union[str, Path],
    model_class: Optional[type] = None,
    device: Optional[torch.device] = None,
) -> Union[torch.nn.Module, dict]:
    """
    Load PyTorch model from checkpoint.

    Parameters
    ----------
    filepath : str or Path
        Path to the model checkpoint.
    model_class : type, optional
        Model class to instantiate. If None, returns checkpoint dict.
    device : torch.device, optional
        Device to load model onto.

    Returns
    -------
    torch.nn.Module or dict
        Loaded model or checkpoint dictionary.

    Examples
    --------
    >>> from nn4psych.models import ActorCritic
    >>> model = load_model('model_weights.pth', ActorCritic)
    """
    filepath = Path(filepath)

    if device is None:
        device = torch.device('cpu')

    checkpoint = torch.load(filepath, map_location=device)

    if model_class is not None and 'model_config' in checkpoint:
        config = checkpoint['model_config']
        model = model_class(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            action_dim=config['action_dim'],
            gain=config.get('gain', 1.5),
            noise=config.get('noise', 0.0),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model

    return checkpoint


def unpickle_state_vector(
    file_dir: Union[str, Path] = "data/rnn_behav/model_params_101000/",
    rnn_param: str = "None",
):
    """
    Unpickle the state vector made by behavior extraction.

    Parameters
    ----------
    file_dir : str or Path
        Directory containing pickled state vectors.
    rnn_param : str
        RNN parameter type: "gamma", "preset", "rollout", "scale", or "combined".

    Returns
    -------
    tuple
        (cp_array, ob_array, model_list) where:
        - cp_array: Change-point condition data
        - ob_array: Oddball condition data
        - model_list: Model metadata dictionary
    """
    import os

    file_dir = Path(file_dir)

    with open(file_dir / f"{rnn_param}_cp_list.pkl", 'rb') as f:
        cp_array = pickle.load(f)

    with open(file_dir / f"{rnn_param}_ob_list.pkl", 'rb') as f:
        ob_array = pickle.load(f)

    with open(file_dir / f"{rnn_param}_dict.pkl", 'rb') as f:
        model_list = pickle.load(f)

    return cp_array, ob_array, model_list
