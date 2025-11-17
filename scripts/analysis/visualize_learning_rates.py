#!/usr/bin/env python3
"""
Visualize Learning Rate Analysis

Creates plots showing learning rate patterns by condition and prediction error.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import (
    LEARNING_RATES_PATH,
    SUMMARY_METRICS_PATH,
    BEHAVIORAL_FIGURES_DIR,
    COLUMN_NAMES,
)


def plot_lr_by_pe(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot learning rate as a function of prediction error.

    Parameters
    ----------
    df : pd.DataFrame
        Learning rate data.
    output_dir : Path
        Directory to save figures.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, condition in enumerate(['change-point', 'oddball']):
        ax = axes[idx]
        subset = df[df[COLUMN_NAMES['condition']] == condition]

        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{condition.capitalize()}')
            continue

        # Bin by prediction error
        bins = np.linspace(20, 150, 20)
        subset['pe_bin'] = pd.cut(subset[COLUMN_NAMES['abs_prediction_error']], bins=bins)

        # Calculate mean LR per bin
        lr_by_bin = subset.groupby('pe_bin')[COLUMN_NAMES['learning_rate']].agg(['mean', 'std', 'count'])
        lr_by_bin = lr_by_bin[lr_by_bin['count'] >= 10]  # Filter low-count bins

        if len(lr_by_bin) > 0:
            x = [interval.mid for interval in lr_by_bin.index]
            y = lr_by_bin['mean'].values
            yerr = lr_by_bin['std'].values / np.sqrt(lr_by_bin['count'].values)

            ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=3, color='steelblue')
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)

        ax.set_xlabel('Absolute Prediction Error')
        ax.set_ylabel('Learning Rate')
        ax.set_title(f'{condition.capitalize()} Condition')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.suptitle('Learning Rate by Prediction Error', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_rate_by_prediction_error.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: learning_rate_by_prediction_error.png")


def plot_lr_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot distribution of learning rates by condition.

    Parameters
    ----------
    df : pd.DataFrame
        Learning rate data.
    output_dir : Path
        Directory to save figures.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in ['change-point', 'oddball']:
        subset = df[df[COLUMN_NAMES['condition']] == condition]
        if len(subset) > 0:
            sns.kdeplot(
                data=subset[COLUMN_NAMES['learning_rate']],
                label=condition.capitalize(),
                ax=ax,
                fill=True,
                alpha=0.3,
            )

    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Learning Rates by Condition')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'learning_rate_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: learning_rate_distribution.png")


def plot_model_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot summary metrics comparing models.

    Parameters
    ----------
    df : pd.DataFrame
        Summary metrics data.
    output_dir : Path
        Directory to save figures.
    """
    if len(df) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Mean LR by condition
    ax = axes[0, 0]
    df_pivot = df.pivot_table(
        values=COLUMN_NAMES['mean_lr'],
        index=COLUMN_NAMES['model_id'],
        columns=COLUMN_NAMES['condition'],
    )
    if len(df_pivot) > 0:
        df_pivot.plot(kind='box', ax=ax)
        ax.set_ylabel('Mean Learning Rate')
        ax.set_title('Mean Learning Rate Distribution')

    # Mean PE by condition
    ax = axes[0, 1]
    df_pivot = df.pivot_table(
        values=COLUMN_NAMES['mean_pe'],
        index=COLUMN_NAMES['model_id'],
        columns=COLUMN_NAMES['condition'],
    )
    if len(df_pivot) > 0:
        df_pivot.plot(kind='box', ax=ax)
        ax.set_ylabel('Mean Prediction Error')
        ax.set_title('Mean Prediction Error Distribution')

    # LR vs PE scatter
    ax = axes[1, 0]
    for condition in df[COLUMN_NAMES['condition']].unique():
        subset = df[df[COLUMN_NAMES['condition']] == condition]
        ax.scatter(
            subset[COLUMN_NAMES['mean_pe']],
            subset[COLUMN_NAMES['mean_lr']],
            label=condition,
            alpha=0.6,
        )
    ax.set_xlabel('Mean Prediction Error')
    ax.set_ylabel('Mean Learning Rate')
    ax.set_title('Learning Rate vs Prediction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Hazard response
    ax = axes[1, 1]
    df.plot.scatter(
        x='total_trials',
        y='n_hazard_events',
        c=COLUMN_NAMES['mean_lr'],
        cmap='viridis',
        ax=ax,
        alpha=0.6,
    )
    ax.set_xlabel('Total Trials')
    ax.set_ylabel('Number of Hazard Events')
    ax.set_title('Hazard Events (colored by mean LR)')

    plt.suptitle('Model Performance Summary', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: model_comparison_summary.png")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Learning Rate Visualization")
    print("=" * 60)

    # Ensure output directory exists
    BEHAVIORAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    if LEARNING_RATES_PATH.exists():
        df_lr = pd.read_csv(LEARNING_RATES_PATH)
        print(f"Loaded {len(df_lr)} learning rate observations")

        plot_lr_by_pe(df_lr, BEHAVIORAL_FIGURES_DIR)
        plot_lr_distribution(df_lr, BEHAVIORAL_FIGURES_DIR)
    else:
        print(f"Learning rates file not found: {LEARNING_RATES_PATH}")

    if SUMMARY_METRICS_PATH.exists():
        df_summary = pd.read_csv(SUMMARY_METRICS_PATH)
        print(f"Loaded {len(df_summary)} summary records")

        plot_model_comparison(df_summary, BEHAVIORAL_FIGURES_DIR)
    else:
        print(f"Summary metrics file not found: {SUMMARY_METRICS_PATH}")

    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
