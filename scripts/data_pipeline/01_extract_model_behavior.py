#!/usr/bin/env python3
"""
Stage 01: Extract Behavioral Data from Trained Models

This script loads trained model weights and extracts behavioral data by
running them through the predictive inference task environments.

Input: Model weight files (.pth) from training runs
Output: Pickled state vectors and CSV summaries in output/behavioral_summary/
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import (
    OUTPUT_DIR,
    BEHAVIORAL_SUMMARY_DIR,
    COLLATED_BEHAVIOR_PATH,
    TRIALS_DATA_PATH,
    MODEL_PARAMS,
    TASK_PARAMS,
    COLUMN_NAMES,
)
from nn4psych.models import ActorCritic
from nn4psych.envs import PIE_CP_OB_v2
from nn4psych.analysis.behavior import extract_behavior
from nn4psych.utils.io import saveload


def extract_single_model_behavior(
    model_path: Path,
    n_epochs: int = 100,
    reset_memory: bool = True,
) -> dict:
    """
    Extract behavioral data from a single trained model.

    Parameters
    ----------
    model_path : Path
        Path to model weights.
    n_epochs : int
        Number of epochs to run.
    reset_memory : bool
        Reset hidden state between epochs.

    Returns
    -------
    dict
        Dictionary containing CP and OB condition data.
    """
    # Load model
    params = MODEL_PARAMS['actor_critic']
    model = ActorCritic(
        input_dim=params['input_dim'],
        hidden_dim=params['hidden_dim'],
        action_dim=params['action_dim'],
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    # Create environments
    env_cp = PIE_CP_OB_v2(**TASK_PARAMS['change_point'])
    env_ob = PIE_CP_OB_v2(**TASK_PARAMS['oddball'])

    # Extract behavior for both conditions
    states_cp = extract_behavior(model, env_cp, n_epochs=n_epochs, reset_memory=reset_memory)
    states_ob = extract_behavior(model, env_ob, n_epochs=n_epochs, reset_memory=reset_memory)

    return {
        'model_path': str(model_path),
        'states_cp': states_cp,
        'states_ob': states_ob,
        'n_epochs': n_epochs,
    }


def states_to_dataframe(states: list, condition: str, model_id: str) -> pd.DataFrame:
    """
    Convert state tuples to long-format DataFrame.

    Parameters
    ----------
    states : list
        List of state tuples per epoch.
    condition : str
        Task condition name.
    model_id : str
        Model identifier.

    Returns
    -------
    pd.DataFrame
        Long-format trial data.
    """
    rows = []
    for epoch_idx, epoch_states in enumerate(states):
        trials, bucket_pos, bag_pos, heli_pos, hazards = epoch_states

        for i in range(len(trials)):
            rows.append({
                COLUMN_NAMES['model_id']: model_id,
                COLUMN_NAMES['epoch']: epoch_idx,
                COLUMN_NAMES['trial']: trials[i],
                COLUMN_NAMES['condition']: condition,
                COLUMN_NAMES['bucket_pos']: bucket_pos[i],
                COLUMN_NAMES['bag_pos']: bag_pos[i],
                COLUMN_NAMES['heli_pos']: heli_pos[i],
                COLUMN_NAMES['hazard']: hazards[i],
            })

    df = pd.DataFrame(rows)

    # Compute prediction error
    df[COLUMN_NAMES['prediction_error']] = df[COLUMN_NAMES['bag_pos']] - df[COLUMN_NAMES['bucket_pos']]
    df[COLUMN_NAMES['abs_prediction_error']] = df[COLUMN_NAMES['prediction_error']].abs()

    return df


def extract_all_models(model_dir: Path, n_epochs: int = 100) -> None:
    """
    Extract behavior from all models in directory.

    Parameters
    ----------
    model_dir : Path
        Directory containing model weights.
    n_epochs : int
        Number of epochs per model.
    """
    model_files = list(model_dir.glob('*.pth'))
    print(f"Found {len(model_files)} model files in {model_dir}")

    if not model_files:
        print("No model files found. Skipping extraction.")
        return

    all_trials_data = []
    all_behavior_data = []

    for model_path in tqdm(model_files, desc="Extracting behavior"):
        try:
            # Extract model ID from filename
            model_id = model_path.stem

            # Extract behavior
            behavior_data = extract_single_model_behavior(model_path, n_epochs=n_epochs)

            # Convert to DataFrame
            df_cp = states_to_dataframe(behavior_data['states_cp'], 'change-point', model_id)
            df_ob = states_to_dataframe(behavior_data['states_ob'], 'oddball', model_id)

            all_trials_data.append(df_cp)
            all_trials_data.append(df_ob)

            # Store raw behavior for pickle
            all_behavior_data.append({
                'model_id': model_id,
                'model_path': str(model_path),
                'states_cp': behavior_data['states_cp'],
                'states_ob': behavior_data['states_ob'],
            })

        except Exception as e:
            print(f"Error processing {model_path}: {e}")
            continue

    if all_trials_data:
        # Save long-format trial data
        df_trials = pd.concat(all_trials_data, ignore_index=True)
        df_trials.to_csv(TRIALS_DATA_PATH, index=False)
        print(f"Saved {len(df_trials)} trials to {TRIALS_DATA_PATH}")

        # Save pickled behavior
        saveload(BEHAVIORAL_SUMMARY_DIR / 'raw_behavior_data', all_behavior_data, 'save')
    else:
        print("No data extracted.")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Stage 01: Extract Model Behavior")
    print("=" * 60)

    # Check for model directories
    from config import PROJECT_ROOT
    model_dirs = list(PROJECT_ROOT.glob('model_params*'))

    if not model_dirs:
        print("No model_params directories found.")
        print("Creating empty output structure...")
        BEHAVIORAL_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        return

    for model_dir in model_dirs:
        print(f"\nProcessing: {model_dir}")
        extract_all_models(model_dir, n_epochs=100)

    print("\nBehavior extraction complete!")


if __name__ == '__main__':
    main()
