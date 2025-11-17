"""
Pytest configuration for validation tests.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_model():
    """Create a sample ActorCritic model for testing."""
    from nn4psych.models import ActorCritic
    return ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)


@pytest.fixture
def sample_env():
    """Create a sample environment for testing."""
    from nn4psych.envs import PIE_CP_OB_v2
    return PIE_CP_OB_v2(condition="change-point", total_trials=10)


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    from nn4psych.training.configs import create_default_config
    config = create_default_config()
    config.training.epochs = 2  # Reduce for testing
    config.task.total_trials = 10
    return config


@pytest.fixture(scope="session")
def validation_seeds():
    """Standard seeds for validation testing."""
    return [42, 123, 456, 789, 1011]
