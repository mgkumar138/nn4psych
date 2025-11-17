"""
NN4Psych Project Configuration

Central configuration file for all paths, parameters, and project settings.
Following data analysis project standards.
"""

from pathlib import Path

# =============================================================================
# DIRECTORY STRUCTURE
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
OUTPUT_DIR = PROJECT_ROOT / 'output'
FIGURE_DIR = PROJECT_ROOT / 'figures'
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
MODELS_DIR = PROJECT_ROOT / 'nn4psych' / 'models'
ENVIRONMENTS_DIR = PROJECT_ROOT / 'nn4psych' / 'envs'
VALIDATION_DIR = PROJECT_ROOT / 'validation'
DOCS_DIR = PROJECT_ROOT / 'docs'

# Create directories if they don't exist
for dir_path in [
    OUTPUT_DIR,
    OUTPUT_DIR / 'behavioral_summary',
    OUTPUT_DIR / 'model_performance',
    OUTPUT_DIR / 'parameter_exploration',
    FIGURE_DIR,
    FIGURE_DIR / 'behavioral_summary',
    FIGURE_DIR / 'model_performance',
    FIGURE_DIR / 'parameter_exploration',
    DOCS_DIR,
    VALIDATION_DIR,
]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# OUTPUT FILE PATHS - Behavioral Data
# =============================================================================

# Behavioral summaries
BEHAVIORAL_SUMMARY_DIR = OUTPUT_DIR / 'behavioral_summary'
COLLATED_BEHAVIOR_PATH = BEHAVIORAL_SUMMARY_DIR / 'collated_model_behavior.csv'
TRIALS_DATA_PATH = BEHAVIORAL_SUMMARY_DIR / 'task_trials_long.csv'
SUMMARY_METRICS_PATH = BEHAVIORAL_SUMMARY_DIR / 'summary_performance_metrics.csv'
LEARNING_RATES_PATH = BEHAVIORAL_SUMMARY_DIR / 'learning_rates_by_condition.csv'

# Model performance
MODEL_PERFORMANCE_DIR = OUTPUT_DIR / 'model_performance'
HYPERPARAMETER_SWEEP_PATH = MODEL_PERFORMANCE_DIR / 'hyperparameter_sweep_results.csv'
MODEL_COMPARISON_PATH = MODEL_PERFORMANCE_DIR / 'model_comparison_metrics.csv'
BEST_MODELS_PATH = MODEL_PERFORMANCE_DIR / 'best_performing_models.csv'

# Parameter exploration
PARAMETER_EXPLORATION_DIR = OUTPUT_DIR / 'parameter_exploration'
GAMMA_SWEEP_PATH = PARAMETER_EXPLORATION_DIR / 'gamma_sweep_results.csv'
ROLLOUT_SWEEP_PATH = PARAMETER_EXPLORATION_DIR / 'rollout_sweep_results.csv'
PRESET_SWEEP_PATH = PARAMETER_EXPLORATION_DIR / 'preset_sweep_results.csv'
SCALE_SWEEP_PATH = PARAMETER_EXPLORATION_DIR / 'scale_sweep_results.csv'

# =============================================================================
# FIGURE OUTPUT PATHS
# =============================================================================

BEHAVIORAL_FIGURES_DIR = FIGURE_DIR / 'behavioral_summary'
PERFORMANCE_FIGURES_DIR = FIGURE_DIR / 'model_performance'
EXPLORATION_FIGURES_DIR = FIGURE_DIR / 'parameter_exploration'

# =============================================================================
# MODEL PARAMETERS - Default Configuration
# =============================================================================

# Actor-Critic Model Parameters
MODEL_PARAMS = {
    'actor_critic': {
        'input_dim': 9,      # 6 observation dims + 2 context dims + 1 reward
        'hidden_dim': 64,    # RNN hidden units
        'action_dim': 3,     # left, right, confirm
        'gain': 1.5,         # RNN weight initialization scaling
        'noise': 0.0,        # Hidden state noise
        'bias': False,       # Use bias in layers
    }
}

# Task Environment Parameters
TASK_PARAMS = {
    'change_point': {
        'condition': 'change-point',
        'total_trials': 200,
        'max_time': 300,
        'train_cond': False,
        'max_displacement': 10.0,
        'reward_size': 5.0,
        'step_cost': 0.0,
        'alpha': 1.0,
        'hazard_rate': 0.125,
    },
    'oddball': {
        'condition': 'oddball',
        'total_trials': 200,
        'max_time': 300,
        'train_cond': False,
        'max_displacement': 10.0,
        'reward_size': 5.0,
        'step_cost': 0.0,
        'alpha': 1.0,
        'hazard_rate': 0.125,
    }
}

# Training Parameters
TRAINING_PARAMS = {
    'default': {
        'epochs': 100,
        'learning_rate': 5e-4,
        'gamma': 0.95,
        'rollout_size': 100,
        'td_noise': 0.0,
        'preset_memory': 0.0,
        'seed': 42,
        'device': 'cpu',
    }
}

# =============================================================================
# HYPERPARAMETER SWEEP VALUES
# =============================================================================

GAMMA_VALUES = [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.25, 0.1]
ROLLOUT_VALUES = [5, 10, 20, 30, 50, 75, 100, 150, 200]
PRESET_VALUES = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
SCALE_VALUES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

# =============================================================================
# DATA COLUMN NAMING CONVENTIONS
# =============================================================================

# Standard column names for consistency
COLUMN_NAMES = {
    # Trial-level data
    'trial': 'trial',
    'epoch': 'epoch',
    'condition': 'condition',
    'bucket_pos': 'bucket_position',
    'bag_pos': 'bag_position',
    'heli_pos': 'helicopter_position',
    'hazard': 'hazard_trigger',

    # Computed metrics
    'prediction_error': 'prediction_error',
    'abs_prediction_error': 'abs_prediction_error',
    'learning_rate': 'learning_rate',
    'update': 'update',

    # Aggregates
    'mean_lr': 'mean_learning_rate',
    'std_lr': 'std_learning_rate',
    'median_lr': 'median_learning_rate',
    'mean_pe': 'mean_prediction_error',
    'std_pe': 'std_prediction_error',

    # Model identifiers
    'model_id': 'model_id',
    'model_path': 'model_path',
    'performance_score': 'performance_score',

    # Hyperparameters
    'gamma': 'gamma',
    'rollout': 'rollout_size',
    'preset': 'preset_memory',
    'scale': 'td_scale',
}

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

# Performance filtering
PERFORMANCE_THRESHOLD = 10.0  # Minimum performance score for inclusion
MIN_PERFORMANCE = 5.0  # Absolute minimum to consider

# Learning rate analysis
LR_PE_THRESHOLD = 20  # Minimum prediction error for LR calculation
LR_CLIP_RANGE = (0.0, 1.0)  # Valid learning rate range

# Smoothing parameters
SMOOTHING_WINDOW = 10  # Window size for uniform filter

# =============================================================================
# VERSIONING
# =============================================================================

PROJECT_VERSION = 'v0.2.0'
DATA_VERSION = 'v1'
ANALYSIS_VERSION = 'v1'

# =============================================================================
# RANDOM SEEDS FOR REPRODUCIBILITY
# =============================================================================

DEFAULT_SEED = 42
VALIDATION_SEEDS = [42, 123, 456, 789, 1011]
