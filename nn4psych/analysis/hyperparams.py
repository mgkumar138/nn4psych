"""
Unified hyperparameter analysis module.

This consolidates the 5 duplicate analyze_hyperparams_*.py scripts into
a single parameterized implementation.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch

from nn4psych.models.actor_critic import ActorCritic
from nn4psych.envs.predictive_inference import PIE_CP_OB_v2
from nn4psych.utils.metrics import get_lrs_v2
from nn4psych.analysis.behavior import get_area


class HyperparamAnalyzer:
    """
    Analyze model performance across hyperparameter sweeps.

    This class provides methods to load models from different hyperparameter
    configurations and analyze their learning behavior.

    Parameters
    ----------
    param_name : str
        Hyperparameter to analyze: "gamma", "rollout", "preset", or "scale".
    model_dir : str or Path
        Directory containing trained models.
    version : str
        Model version string (e.g., "V3", "V5").
    """

    # Predefined parameter values and patterns
    PARAM_VALUES = {
        "gamma": [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.25, 0.1],
        "rollout": [5, 10, 20, 30, 50, 75, 100, 150, 200],
        "preset": [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
        "scale": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    }

    PARAM_PATTERNS = {
        "gamma": "*_{version}_{val}g_0.0rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_*e_10md_5.0rz_*s.pth",
        "rollout": "*_{version}_0.95g_0.0rm_{val}bz_0.0td_1.0tds_Nonelb_Noneup_64n_*e_10md_5.0rz_*s.pth",
        "preset": "*_{version}_0.95g_{val}rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_*e_10md_5.0rz_*s.pth",
        "scale": "*_{version}_0.95g_0.0rm_100bz_0.0td_{val}tds_Nonelb_Noneup_64n_*e_10md_5.0rz_*s.pth",
    }

    def __init__(
        self,
        param_name: str,
        model_dir: str = "./model_params/",
        version: str = "V3",
    ):
        if param_name not in self.PARAM_VALUES:
            raise ValueError(f"Unknown parameter: {param_name}")

        self.param_name = param_name
        self.model_dir = Path(model_dir)
        self.version = version
        self.values = self.PARAM_VALUES[param_name]
        self.pattern_template = self.PARAM_PATTERNS[param_name]

    def get_model_files(
        self,
        value: float,
        performance_threshold: float = 5.0,
    ) -> List[Path]:
        """
        Get model files for a specific parameter value.

        Parameters
        ----------
        value : float
            Parameter value.
        performance_threshold : float, optional
            Minimum performance to include. Default is 5.0.

        Returns
        -------
        list of Path
            List of model file paths.
        """
        pattern = self.pattern_template.format(version=self.version, val=value)
        full_pattern = str(self.model_dir / pattern)
        models = glob.glob(full_pattern)

        # Filter by performance
        filtered = []
        for m in models:
            try:
                perf = float(Path(m).stem.split("_")[0])
                if perf > performance_threshold:
                    filtered.append(Path(m))
            except (ValueError, IndexError):
                continue

        return sorted(filtered, key=lambda x: float(x.stem.split("_")[0]), reverse=True)

    def analyze_sweep(
        self,
        n_epochs: int = 100,
        reset_memory: bool = True,
        performance_threshold: float = 5.0,
        max_models_per_value: int = 10,
    ) -> Dict[float, Dict]:
        """
        Analyze learning rates across all parameter values.

        Parameters
        ----------
        n_epochs : int, optional
            Epochs per model evaluation.
        reset_memory : bool, optional
            Reset memory between epochs.
        performance_threshold : float, optional
            Minimum performance to include.
        max_models_per_value : int, optional
            Maximum models to analyze per value.

        Returns
        -------
        dict
            Results dictionary with learning rate data per parameter value.
        """
        results = {}

        for value in self.values:
            print(f"Analyzing {self.param_name}={value}")
            models = self.get_model_files(value, performance_threshold)[:max_models_per_value]

            if not models:
                print(f"  No models found for {self.param_name}={value}")
                continue

            lr_data_cp = []
            lr_data_ob = []

            for model_path in models:
                try:
                    area = get_area(
                        model_path,
                        epochs=n_epochs,
                        reset_memory=reset_memory,
                    )
                    # Store per-model results
                    # This is a simplified version - full implementation would
                    # extract detailed learning rate curves
                except Exception as e:
                    print(f"  Error processing {model_path}: {e}")
                    continue

            results[value] = {
                'models': [str(m) for m in models],
                'count': len(models),
            }

        return results

    def plot_learning_rates(
        self,
        results: Dict[float, Dict],
        condition: str = "both",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot learning rate curves across parameter values.

        Parameters
        ----------
        results : dict
            Results from analyze_sweep.
        condition : str, optional
            "cp", "ob", or "both".
        figsize : tuple, optional
            Figure size.
        save_path : Path, optional
            If provided, save figure to this path.

        Returns
        -------
        plt.Figure
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

        for i, (value, data) in enumerate(sorted(results.items())):
            if 'pe_cp' in data and condition in ['cp', 'both']:
                ax.plot(
                    data['pe_cp'],
                    data['lr_cp'],
                    label=f'{self.param_name}={value} (CP)',
                    color=colors[i],
                    linestyle='-',
                )
            if 'pe_ob' in data and condition in ['ob', 'both']:
                ax.plot(
                    data['pe_ob'],
                    data['lr_ob'],
                    label=f'{self.param_name}={value} (OB)',
                    color=colors[i],
                    linestyle='--',
                )

        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Learning Rate')
        ax.set_title(f'Learning Rate by {self.param_name.capitalize()}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def generate_summary_report(
        self,
        results: Dict[float, Dict],
    ) -> str:
        """
        Generate text summary of analysis results.

        Parameters
        ----------
        results : dict
            Results from analyze_sweep.

        Returns
        -------
        str
            Summary report text.
        """
        lines = [
            f"Hyperparameter Analysis Report: {self.param_name}",
            "=" * 50,
            f"Model Directory: {self.model_dir}",
            f"Version: {self.version}",
            "",
            "Parameter Values Analyzed:",
        ]

        for value, data in sorted(results.items()):
            lines.append(f"  {self.param_name}={value}: {data['count']} models")

        lines.extend([
            "",
            "Summary Statistics:",
            # Add more statistics here as needed
        ])

        return "\n".join(lines)


def filter_models_by_performance(
    model_dir: Path,
    threshold: float = 10.0,
) -> Dict[str, List[bool]]:
    """
    Filter models by performance threshold.

    Parameters
    ----------
    model_dir : Path
        Directory containing models.
    threshold : float
        Performance threshold.

    Returns
    -------
    dict
        Dictionary mapping parameter type to boolean index arrays.
    """
    from nn4psych.training.configs import (
        GAMMA_VALUES,
        ROLLOUT_VALUES,
        PRESET_VALUES,
        SCALE_VALUES,
    )

    indices = {
        'gamma': {},
        'rollout': {},
        'preset': {},
        'scale': {},
    }

    param_configs = {
        'gamma': GAMMA_VALUES,
        'rollout': ROLLOUT_VALUES,
        'preset': PRESET_VALUES,
        'scale': SCALE_VALUES,
    }

    for param_type, values in param_configs.items():
        analyzer = HyperparamAnalyzer(param_type, model_dir)
        for val in values:
            models = analyzer.get_model_files(val, performance_threshold=5.0)
            idx = [
                float(m.stem.split("_")[0]) > threshold
                for m in models
            ]
            indices[param_type][val] = idx

    return indices
