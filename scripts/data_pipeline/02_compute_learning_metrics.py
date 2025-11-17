#!/usr/bin/env python3
"""
Stage 02: Compute Learning Rate Metrics

This script processes the extracted behavioral data to compute learning rates,
prediction errors, and other performance metrics.

Input: task_trials_long.csv from Stage 01
Output: learning_rates_by_condition.csv, summary_performance_metrics.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import (
    BEHAVIORAL_SUMMARY_DIR,
    TRIALS_DATA_PATH,
    LEARNING_RATES_PATH,
    SUMMARY_METRICS_PATH,
    COLUMN_NAMES,
    LR_PE_THRESHOLD,
    LR_CLIP_RANGE,
)


def compute_learning_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute learning rates from trial data.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level data with bucket and bag positions.

    Returns
    -------
    pd.DataFrame
        DataFrame with learning rate calculations.
    """
    # Group by model, epoch, condition
    grouped = df.groupby([
        COLUMN_NAMES['model_id'],
        COLUMN_NAMES['epoch'],
        COLUMN_NAMES['condition'],
    ])

    lr_rows = []
    for (model_id, epoch, condition), group in grouped:
        group = group.sort_values(COLUMN_NAMES['trial'])

        bucket_pos = group[COLUMN_NAMES['bucket_pos']].values
        bag_pos = group[COLUMN_NAMES['bag_pos']].values

        # Compute updates and prediction errors
        prediction_error = (bag_pos - bucket_pos)[:-1]
        update = np.diff(bucket_pos)

        # Filter non-zero prediction errors
        valid_idx = prediction_error != 0
        pe_valid = prediction_error[valid_idx]
        update_valid = update[valid_idx]

        if len(pe_valid) == 0:
            continue

        # Calculate learning rate
        learning_rate = update_valid / pe_valid

        # Filter by threshold and clip
        abs_pe = np.abs(pe_valid)
        threshold_idx = abs_pe > LR_PE_THRESHOLD
        pe_filtered = abs_pe[threshold_idx]
        lr_filtered = np.clip(learning_rate[threshold_idx], *LR_CLIP_RANGE)

        for i in range(len(pe_filtered)):
            lr_rows.append({
                COLUMN_NAMES['model_id']: model_id,
                COLUMN_NAMES['epoch']: epoch,
                COLUMN_NAMES['condition']: condition,
                COLUMN_NAMES['abs_prediction_error']: pe_filtered[i],
                COLUMN_NAMES['learning_rate']: lr_filtered[i],
            })

    return pd.DataFrame(lr_rows)


def compute_summary_metrics(df_trials: pd.DataFrame, df_lr: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary metrics per model.

    Parameters
    ----------
    df_trials : pd.DataFrame
        Trial-level data.
    df_lr : pd.DataFrame
        Learning rate data.

    Returns
    -------
    pd.DataFrame
        Summary statistics per model and condition.
    """
    summary_rows = []

    # Group by model and condition
    for (model_id, condition), group in df_trials.groupby([
        COLUMN_NAMES['model_id'],
        COLUMN_NAMES['condition'],
    ]):
        # Basic trial metrics
        mean_pe = group[COLUMN_NAMES['abs_prediction_error']].mean()
        std_pe = group[COLUMN_NAMES['abs_prediction_error']].std()
        median_pe = group[COLUMN_NAMES['abs_prediction_error']].median()

        # Learning rate metrics
        lr_subset = df_lr[
            (df_lr[COLUMN_NAMES['model_id']] == model_id) &
            (df_lr[COLUMN_NAMES['condition']] == condition)
        ]

        if len(lr_subset) > 0:
            mean_lr = lr_subset[COLUMN_NAMES['learning_rate']].mean()
            std_lr = lr_subset[COLUMN_NAMES['learning_rate']].std()
            median_lr = lr_subset[COLUMN_NAMES['learning_rate']].median()
        else:
            mean_lr = std_lr = median_lr = np.nan

        # Hazard response
        n_hazards = group[COLUMN_NAMES['hazard']].sum()
        total_trials = len(group)

        summary_rows.append({
            COLUMN_NAMES['model_id']: model_id,
            COLUMN_NAMES['condition']: condition,
            COLUMN_NAMES['mean_pe']: mean_pe,
            COLUMN_NAMES['std_pe']: std_pe,
            'median_prediction_error': median_pe,
            COLUMN_NAMES['mean_lr']: mean_lr,
            COLUMN_NAMES['std_lr']: std_lr,
            COLUMN_NAMES['median_lr']: median_lr,
            'n_hazard_events': n_hazards,
            'total_trials': total_trials,
            'hazard_rate': n_hazards / total_trials if total_trials > 0 else 0,
        })

    return pd.DataFrame(summary_rows)


def main():
    """Main entry point."""
    print("=" * 60)
    print("Stage 02: Compute Learning Rate Metrics")
    print("=" * 60)

    # Load trial data
    if not TRIALS_DATA_PATH.exists():
        print(f"Trial data not found at {TRIALS_DATA_PATH}")
        print("Run Stage 01 first to extract model behavior.")
        return

    df_trials = pd.read_csv(TRIALS_DATA_PATH)
    print(f"Loaded {len(df_trials)} trials from {TRIALS_DATA_PATH}")

    # Compute learning rates
    print("\nComputing learning rates...")
    df_lr = compute_learning_rates(df_trials)
    df_lr.to_csv(LEARNING_RATES_PATH, index=False)
    print(f"Saved {len(df_lr)} learning rate observations to {LEARNING_RATES_PATH}")

    # Compute summary metrics
    print("\nComputing summary metrics...")
    df_summary = compute_summary_metrics(df_trials, df_lr)
    df_summary.to_csv(SUMMARY_METRICS_PATH, index=False)
    print(f"Saved summary for {len(df_summary)} model-condition pairs to {SUMMARY_METRICS_PATH}")

    print("\nMetrics computation complete!")


if __name__ == '__main__':
    main()
