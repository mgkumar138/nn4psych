"""
Configuration management using dataclasses.

This module provides structured configuration objects that replace hard-coded
parameters throughout the codebase, enabling reproducible experiments.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import yaml
import json


@dataclass
class ModelConfig:
    """
    Configuration for the ActorCritic model.

    Attributes
    ----------
    input_dim : int
        Input feature dimension. Default is 9 (6 obs + 2 context + 1 reward).
    hidden_dim : int
        Number of RNN hidden units. Default is 64.
    action_dim : int
        Number of possible actions. Default is 3.
    gain : float
        RNN weight initialization gain. Default is 1.5.
    noise : float
        Hidden state noise variance. Default is 0.0.
    bias : bool
        Whether to include bias terms. Default is False.
    """

    input_dim: int = 9
    hidden_dim: int = 64
    action_dim: int = 3
    gain: float = 1.5
    noise: float = 0.0
    bias: bool = False


@dataclass
class TaskConfig:
    """
    Configuration for the predictive inference task environment.

    Attributes
    ----------
    condition : str
        Task condition: "change-point" or "oddball".
    total_trials : int
        Number of trials per epoch.
    max_time : int
        Maximum time steps per trial.
    train_cond : bool
        Whether helicopter position is visible.
    max_displacement : float
        Maximum bucket movement per action.
    reward_size : float
        SD for Gaussian reward function.
    step_cost : float
        Per-step penalty.
    alpha : float
        Velocity smoothing factor.
    """

    condition: str = "change-point"
    total_trials: int = 200
    max_time: int = 300
    train_cond: bool = False
    max_displacement: float = 10.0
    reward_size: float = 5.0
    step_cost: float = 0.0
    alpha: float = 1.0


@dataclass
class TrainingConfig:
    """
    Configuration for training parameters.

    Attributes
    ----------
    epochs : int
        Number of training epochs.
    learning_rate : float
        Optimizer learning rate.
    gamma : float
        Discount factor for returns.
    rollout_size : int
        Number of steps to collect before update.
    td_noise : float
        Temporal difference noise.
    preset_memory : float
        Initial hidden state value.
    td_lower_bound : float or None
        Lower bound for TD updates.
    td_upper_bound : float or None
        Upper bound for TD updates.
    td_scale : float
        TD scaling factor.
    seed : int
        Random seed for reproducibility.
    device : str
        PyTorch device (cpu or cuda).
    save_dir : str
        Directory to save model weights.
    save_frequency : int
        Save model every N epochs.
    """

    epochs: int = 100
    learning_rate: float = 5e-4
    gamma: float = 0.95
    rollout_size: int = 100
    td_noise: float = 0.0
    preset_memory: float = 0.0
    td_lower_bound: Optional[float] = None
    td_upper_bound: Optional[float] = None
    td_scale: float = 1.0
    seed: int = 42
    device: str = "cpu"
    save_dir: str = "model_params"
    save_frequency: int = 10


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.

    Attributes
    ----------
    name : str
        Experiment name/identifier.
    model : ModelConfig
        Model configuration.
    task : TaskConfig
        Task configuration.
    training : TrainingConfig
        Training configuration.
    description : str
        Optional experiment description.
    tags : list
        Optional tags for organizing experiments.
    """

    name: str = "experiment"
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    description: str = ""
    tags: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_yaml(self, filepath: Optional[Path] = None) -> str:
        """
        Export configuration to YAML format.

        Parameters
        ----------
        filepath : Path, optional
            If provided, save to file.

        Returns
        -------
        str
            YAML string representation.
        """
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if filepath is not None:
            with open(filepath, 'w') as f:
                f.write(yaml_str)
        return yaml_str

    def to_json(self, filepath: Optional[Path] = None) -> str:
        """
        Export configuration to JSON format.

        Parameters
        ----------
        filepath : Path, optional
            If provided, save to file.

        Returns
        -------
        str
            JSON string representation.
        """
        json_str = json.dumps(self.to_dict(), indent=2)
        if filepath is not None:
            with open(filepath, 'w') as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_yaml(cls, filepath: Path) -> "ExperimentConfig":
        """
        Load configuration from YAML file.

        Parameters
        ----------
        filepath : Path
            Path to YAML file.

        Returns
        -------
        ExperimentConfig
            Loaded configuration.
        """
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, filepath: Path) -> "ExperimentConfig":
        """
        Load configuration from JSON file.

        Parameters
        ----------
        filepath : Path
            Path to JSON file.

        Returns
        -------
        ExperimentConfig
            Loaded configuration.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create configuration from dictionary."""
        return cls(
            name=data.get('name', 'experiment'),
            model=ModelConfig(**data.get('model', {})),
            task=TaskConfig(**data.get('task', {})),
            training=TrainingConfig(**data.get('training', {})),
            description=data.get('description', ''),
            tags=data.get('tags', []),
        )

    def get_filename(self) -> str:
        """
        Generate a descriptive filename based on configuration.

        Returns
        -------
        str
            Filename string encoding key hyperparameters.
        """
        return (
            f"V5_{self.training.gamma}g_{self.training.preset_memory}rm_"
            f"{self.training.rollout_size}bz_{self.training.td_noise}td_"
            f"{self.training.td_scale}tds_{self.training.td_lower_bound}lb_"
            f"{self.training.td_upper_bound}up_{self.model.hidden_dim}n_"
            f"{self.training.epochs}e_{self.task.max_displacement}md_"
            f"{self.task.reward_size}rz_{self.training.seed}s"
        )


# Predefined hyperparameter sweep values
GAMMA_VALUES = [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.25, 0.1]
ROLLOUT_VALUES = [5, 10, 20, 30, 50, 75, 100, 150, 200]
PRESET_VALUES = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
SCALE_VALUES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def create_default_config() -> ExperimentConfig:
    """
    Create default experiment configuration.

    Returns
    -------
    ExperimentConfig
        Default configuration matching pretrain_rnn_with_heli_v5.py defaults.
    """
    return ExperimentConfig(
        name="default_experiment",
        model=ModelConfig(
            input_dim=9,
            hidden_dim=64,
            action_dim=3,
            gain=1.5,
            noise=0.0,
            bias=False,
        ),
        task=TaskConfig(
            condition="change-point",
            total_trials=200,
            max_time=300,
            train_cond=False,
            max_displacement=10.0,
            reward_size=5.0,
            step_cost=0.0,
            alpha=1.0,
        ),
        training=TrainingConfig(
            epochs=100,
            learning_rate=5e-4,
            gamma=0.95,
            rollout_size=100,
            td_noise=0.0,
            preset_memory=0.0,
            td_lower_bound=None,
            td_upper_bound=None,
            td_scale=1.0,
            seed=42,
            device="cpu",
            save_dir="model_params",
            save_frequency=10,
        ),
        description="Default configuration for predictive inference task",
        tags=["default", "baseline"],
    )
