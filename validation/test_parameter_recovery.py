"""
Parameter Recovery Tests

Validates that model fitting procedures can recover known parameters.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nn4psych.models import ActorCritic
from nn4psych.envs import PIE_CP_OB_v2


class TestParameterRecovery:
    """Test parameter recovery capabilities."""

    @pytest.mark.parametrize("hidden_dim", [32, 64, 128])
    def test_model_saves_and_loads_correctly(self, hidden_dim, tmp_path):
        """Test that model parameters are preserved through save/load cycle."""
        # Create model
        model = ActorCritic(input_dim=9, hidden_dim=hidden_dim, action_dim=3)

        # Get original parameters
        original_params = {k: v.clone() for k, v in model.state_dict().items()}

        # Save model
        save_path = tmp_path / "test_model.pth"
        torch.save(model.state_dict(), save_path)

        # Load into new model
        new_model = ActorCritic(input_dim=9, hidden_dim=hidden_dim, action_dim=3)
        new_model.load_state_dict(torch.load(save_path))

        # Compare parameters
        for key in original_params:
            assert torch.allclose(original_params[key], new_model.state_dict()[key]), \
                f"Parameter {key} not recovered correctly"

    @pytest.mark.slow
    def test_behavior_is_deterministic_with_seed(self):
        """Test that same seed produces same behavior."""
        from nn4psych.analysis.behavior import extract_behavior

        # Set seed
        torch.manual_seed(42)
        np.random.seed(42)

        model = ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)
        env = PIE_CP_OB_v2(condition="change-point", total_trials=10)

        # First run
        states1 = extract_behavior(model, env, n_epochs=2)

        # Reset seed
        torch.manual_seed(42)
        np.random.seed(42)

        model2 = ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)
        env2 = PIE_CP_OB_v2(condition="change-point", total_trials=10)

        # Second run
        states2 = extract_behavior(model2, env2, n_epochs=2)

        # Compare results
        for epoch in range(len(states1)):
            for i in range(5):  # Compare all state components
                assert np.allclose(states1[epoch][i], states2[epoch][i]), \
                    f"Epoch {epoch}, component {i} not deterministic"

    def test_learning_rate_calculation_edge_cases(self):
        """Test learning rate calculation handles edge cases."""
        from nn4psych.utils.metrics import get_lrs_v2

        # Case 1: Zero prediction errors
        states = (
            np.arange(10),  # trials
            np.array([100] * 10),  # bucket (no movement)
            np.array([100] * 10),  # bag (same as bucket)
            np.array([100] * 10),  # helicopter
            np.zeros(10),  # hazards
        )
        pe, lr = get_lrs_v2(states, threshold=20)
        # Should handle gracefully without errors
        assert len(pe) == len(lr)

    def test_config_serialization_roundtrip(self, tmp_path):
        """Test that config can be saved and loaded without data loss."""
        from nn4psych.training.configs import ExperimentConfig, create_default_config

        original_config = create_default_config()
        original_config.training.gamma = 0.85
        original_config.model.hidden_dim = 128

        # Save to YAML
        yaml_path = tmp_path / "config.yaml"
        original_config.to_yaml(yaml_path)

        # Load back
        loaded_config = ExperimentConfig.from_yaml(yaml_path)

        # Compare
        assert loaded_config.training.gamma == 0.85
        assert loaded_config.model.hidden_dim == 128
        assert loaded_config.name == original_config.name

    @pytest.mark.parametrize("condition", ["change-point", "oddball"])
    def test_environment_reset_is_consistent(self, condition):
        """Test that environment reset produces valid observations."""
        env = PIE_CP_OB_v2(condition=condition, total_trials=10)

        for _ in range(5):
            obs, done = env.reset()

            # Check observation dimensions
            assert len(obs) == 6, "Observation should have 6 dimensions"

            # Check done flag
            assert done is False, "Should not be done after reset"

            # Check observation bounds
            assert obs[1] > 0, "Bucket position should be positive"
            assert obs[1] < 302, "Bucket position should be < 302"


@pytest.mark.integration
class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_behavior_extraction_pipeline(self, sample_model, sample_env):
        """Test complete behavior extraction pipeline."""
        from nn4psych.analysis.behavior import extract_behavior

        states = extract_behavior(sample_model, sample_env, n_epochs=2)

        assert len(states) == 2, "Should have 2 epochs of data"
        for epoch_states in states:
            assert len(epoch_states) == 5, "Each epoch should have 5 state components"
            trials, bucket, bag, heli, hazards = epoch_states
            assert len(trials) == 10, "Should have 10 trials per epoch"
