#!/usr/bin/env python3
"""
Stage 03: Analyze Hyperparameter Sweeps

This script analyzes model performance across different hyperparameter values.

Input: Model files with encoded hyperparameters in filenames
Output: Sweep results CSV files in output/parameter_exploration/
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from glob import glob

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import (
    PROJECT_ROOT,
    PARAMETER_EXPLORATION_DIR,
    GAMMA_SWEEP_PATH,
    ROLLOUT_SWEEP_PATH,
    PRESET_SWEEP_PATH,
    SCALE_SWEEP_PATH,
    GAMMA_VALUES,
    ROLLOUT_VALUES,
    PRESET_VALUES,
    SCALE_VALUES,
    PERFORMANCE_THRESHOLD,
    COLUMN_NAMES,
)


def parse_model_filename(filename: str) -> dict:
    """
    Parse hyperparameters from model filename.

    Expected format: {perf}_{version}_{gamma}g_{preset}rm_{rollout}bz_...

    Parameters
    ----------
    filename : str
        Model filename without extension.

    Returns
    -------
    dict
        Parsed hyperparameters.
    """
    parts = filename.split('_')

    try:
        params = {
            'performance_score': float(parts[0]),
            'version': parts[1],
        }

        # Extract specific parameters
        for part in parts[2:]:
            if part.endswith('g'):
                params['gamma'] = float(part[:-1])
            elif part.endswith('rm'):
                params['preset_memory'] = float(part[:-2])
            elif part.endswith('bz'):
                params['rollout_size'] = int(part[:-2])
            elif part.endswith('tds'):
                params['td_scale'] = float(part[:-3])
            elif part.endswith('n'):
                params['hidden_dim'] = int(part[:-1])
            elif part.endswith('e'):
                params['epochs'] = int(part[:-1])
            elif part.endswith('s'):
                params['seed'] = int(part[:-1])

        return params
    except (ValueError, IndexError):
        return {}


def analyze_sweep(
    model_dir: Path,
    param_name: str,
    param_values: list,
) -> pd.DataFrame:
    """
    Analyze performance across parameter values.

    Parameters
    ----------
    model_dir : Path
        Directory containing models.
    param_name : str
        Parameter to analyze.
    param_values : list
        Values to analyze.

    Returns
    -------
    pd.DataFrame
        Sweep results.
    """
    rows = []

    model_files = list(model_dir.glob('*.pth'))
    print(f"Found {len(model_files)} model files")

    for model_path in model_files:
        params = parse_model_filename(model_path.stem)

        if not params:
            continue

        if params.get('performance_score', 0) < PERFORMANCE_THRESHOLD:
            continue

        if param_name in params:
            rows.append({
                'model_path': str(model_path),
                'model_id': model_path.stem,
                **params,
            })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df

    # Add summary statistics
    summary_rows = []
    for value in param_values:
        subset = df[df[param_name] == value]
        if len(subset) > 0:
            summary_rows.append({
                param_name: value,
                'n_models': len(subset),
                'mean_performance': subset['performance_score'].mean(),
                'std_performance': subset['performance_score'].std(),
                'max_performance': subset['performance_score'].max(),
                'min_performance': subset['performance_score'].min(),
            })

    return pd.DataFrame(summary_rows)


def main():
    """Main entry point."""
    print("=" * 60)
    print("Stage 03: Analyze Hyperparameter Sweeps")
    print("=" * 60)

    # Find model directories
    model_dirs = list(PROJECT_ROOT.glob('model_params*'))

    if not model_dirs:
        print("No model directories found. Creating empty output files.")
        PARAMETER_EXPLORATION_DIR.mkdir(parents=True, exist_ok=True)
        return

    for model_dir in model_dirs:
        print(f"\nAnalyzing: {model_dir}")

        # Analyze each parameter
        sweeps = {
            'gamma': (GAMMA_VALUES, GAMMA_SWEEP_PATH),
            'rollout_size': (ROLLOUT_VALUES, ROLLOUT_SWEEP_PATH),
            'preset_memory': (PRESET_VALUES, PRESET_SWEEP_PATH),
            'td_scale': (SCALE_VALUES, SCALE_SWEEP_PATH),
        }

        for param_name, (values, output_path) in sweeps.items():
            print(f"\n  Analyzing {param_name}...")
            df_sweep = analyze_sweep(model_dir, param_name, values)

            if len(df_sweep) > 0:
                df_sweep.to_csv(output_path, index=False)
                print(f"  Saved {len(df_sweep)} results to {output_path}")
            else:
                print(f"  No results for {param_name}")

    print("\nHyperparameter analysis complete!")


if __name__ == '__main__':
    main()
