# NN4Psych: Neural Networks for Psychological Modeling

A modular Python package for training and analyzing RNN actor-critic models on predictive inference tasks, based on the change-point and oddball paradigms from Nassar et al. 2021.

## Features

- **Modular Architecture**: Clean separation of models, environments, training, and analysis
- **Configuration-Driven**: YAML/JSON configuration files for reproducible experiments
- **Consolidated Codebase**: Single source of truth for all core components (no more duplicates!)
- **Pip-Installable**: Proper Python package structure with `pyproject.toml`
- **Extensible**: Easy to add new models, environments, or analysis methods

## Installation

```bash
# Install in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[dev]"        # Development tools
pip install -e ".[bayesian]"   # Bayesian modeling
pip install -e ".[all]"        # All optional dependencies
```

## Quick Start

### Training a Model

```python
from nn4psych.models import ActorCritic
from nn4psych.envs import PIE_CP_OB_v2
from nn4psych.training.configs import create_default_config

# Create configuration
config = create_default_config()

# Create model and environment
model = ActorCritic.from_config(config.model)
env = PIE_CP_OB_v2.from_config(config.task)

# Train (see scripts/train_example.py for full training loop)
```

### Using Configuration Files

```python
from nn4psych.training.configs import ExperimentConfig
from pathlib import Path

# Load from YAML
config = ExperimentConfig.from_yaml(Path("nn4psych/configs/default.yaml"))

# Save to YAML
config.to_yaml(Path("my_experiment.yaml"))

# Get standardized filename
filename = config.get_filename()
```

### Analyzing Behavior

```python
from nn4psych.analysis.behavior import extract_behavior
from nn4psych.utils.io import load_model

# Load trained model
model = load_model("model_weights.pth", ActorCritic)

# Extract behavioral data
states = extract_behavior(model, env, n_epochs=100)

# Analyze learning rates
from nn4psych.utils.metrics import get_lrs_v2
pe, lr = get_lrs_v2(states[0])  # First epoch
```

### Hyperparameter Analysis

```bash
# Analyze gamma sweep
python scripts/analyze_hyperparams_unified.py --param gamma --model_dir ./model_params/

# Analyze rollout sweep
python scripts/analyze_hyperparams_unified.py --param rollout --model_dir ./model_params/
```

## Package Structure

```
nn4psych/
├── nn4psych/                 # Main package
│   ├── models/              # Neural network models
│   │   └── actor_critic.py  # Consolidated ActorCritic (was 8 copies!)
│   ├── envs/                # Task environments
│   │   └── predictive_inference.py  # PIE_CP_OB_v2
│   ├── training/            # Training infrastructure
│   │   └── configs.py       # Configuration dataclasses
│   ├── analysis/            # Analysis tools
│   │   ├── behavior.py      # Behavior extraction
│   │   └── hyperparams.py   # Unified hyperparam analysis (was 5 scripts!)
│   ├── utils/               # Utilities
│   │   ├── io.py           # I/O functions
│   │   ├── metrics.py      # Learning rate calculations
│   │   └── plotting.py     # Visualization
│   └── configs/             # Default configs
│       └── default.yaml
├── scripts/                  # Example scripts
│   ├── train_example.py     # Training example
│   └── analyze_hyperparams_unified.py
├── tests/                    # Unit tests
├── archive/                  # Archived legacy code
├── pyproject.toml           # Package configuration
└── README.md
```

## Migration from Old Structure

The new package consolidates and improves upon the previous structure:

| Old | New | Improvement |
|-----|-----|-------------|
| 8 copies of `ActorCritic` | 1 in `nn4psych/models/` | Single source of truth |
| 5 `analyze_hyperparams_*.py` | 1 `HyperparamAnalyzer` class | Parameterized, no duplication |
| `utils.py` + `utils_funcs.py` | `nn4psych/utils/` modules | Organized by purpose |
| Hard-coded parameters | `ExperimentConfig` dataclasses | Reproducible, configurable |
| v0, v1, v2, toy directories | Archived, canonical in package | Clear active codebase |

### Importing Old Code

```python
# OLD
from utils_funcs import ActorCritic, get_lrs_v2
from tasks import PIE_CP_OB_v2

# NEW
from nn4psych.models import ActorCritic
from nn4psych.utils.metrics import get_lrs_v2
from nn4psych.envs import PIE_CP_OB_v2
```

## Configuration Reference

### ModelConfig
- `input_dim`: Input feature dimension (default: 9)
- `hidden_dim`: RNN hidden units (default: 64)
- `action_dim`: Number of actions (default: 3)
- `gain`: Weight initialization gain (default: 1.5)
- `noise`: Hidden state noise (default: 0.0)
- `bias`: Use bias terms (default: False)

### TaskConfig
- `condition`: "change-point" or "oddball"
- `total_trials`: Trials per epoch (default: 200)
- `max_time`: Max timesteps per trial (default: 300)
- `max_displacement`: Bucket movement (default: 10.0)
- `reward_size`: Gaussian reward SD (default: 5.0)

### TrainingConfig
- `epochs`: Training epochs (default: 100)
- `learning_rate`: Adam LR (default: 5e-4)
- `gamma`: Discount factor (default: 0.95)
- `rollout_size`: Steps before update (default: 100)
- `seed`: Random seed (default: 42)

## Hyperparameter Sweep Values

Standard sweep ranges used in experiments:

- **gamma** (discount): [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.25, 0.1]
- **rollout** (batch size): [5, 10, 20, 30, 50, 75, 100, 150, 200]
- **preset** (memory init): [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
- **scale** (TD scale): [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

## Legacy Code

Previous implementations are preserved in the `archive/` directory:
- `v0/`: Early JAX implementations
- `v1/`: JAX actor-critic variants
- `v2/`: PyTorch variants (simpler)
- `toy/`: Minimal prototype

These are kept for reference but are not maintained.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black nn4psych/

# Type checking
mypy nn4psych/
```

## References

- Nassar et al. (2021) - Change-point and oddball learning paradigms
- Wang et al. (2018) - RNN actor-critic foundations

## License

MIT License
