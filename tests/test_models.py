"""
Tests for nn4psych models.
"""

import pytest
import torch

from nn4psych.models.actor_critic import ActorCritic


class TestActorCritic:
    """Test ActorCritic model functionality."""

    def test_initialization(self):
        """Test model initialization with default parameters."""
        model = ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)

        assert model.input_dim == 9
        assert model.hidden_dim == 64
        assert model.action_dim == 3
        assert model.gain == 1.5
        assert model.noise == 0.0

    def test_forward_pass(self):
        """Test forward pass produces correct shapes."""
        model = ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)

        batch_size = 1
        seq_len = 1
        x = torch.randn(batch_size, seq_len, 9)
        hx = torch.zeros(1, batch_size, 64)

        actor_logits, critic_value, new_hx = model(x, hx)

        assert actor_logits.shape == (batch_size, 3)
        assert critic_value.shape == (batch_size, 1)
        assert new_hx.shape == (1, batch_size, 64)

    def test_get_initial_hidden(self):
        """Test initial hidden state generation."""
        model = ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)

        h = model.get_initial_hidden(batch_size=2)

        assert h.shape == (1, 2, 64)
        assert torch.all(h == 0)

    def test_reset_hidden_with_preset(self):
        """Test resetting hidden state with preset value."""
        model = ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)

        preset_val = 0.5
        h = model.reset_hidden(batch_size=1, preset_value=preset_val)

        assert h.shape == (1, 1, 64)
        assert torch.allclose(h, torch.full_like(h, preset_val))

    def test_different_hidden_dims(self):
        """Test model with different hidden dimensions."""
        for hidden_dim in [32, 64, 128, 256]:
            model = ActorCritic(input_dim=9, hidden_dim=hidden_dim, action_dim=3)
            x = torch.randn(1, 1, 9)
            hx = torch.zeros(1, 1, hidden_dim)

            actor_logits, critic_value, new_hx = model(x, hx)

            assert actor_logits.shape == (1, 3)
            assert critic_value.shape == (1, 1)
            assert new_hx.shape == (1, 1, hidden_dim)

    def test_weight_initialization(self):
        """Test that weights are properly initialized."""
        model = ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)

        # Check that weights are not all zeros
        for name, param in model.named_parameters():
            if 'weight' in name:
                assert not torch.all(param == 0), f"{name} is all zeros"

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)
        x = torch.randn(1, 1, 9, requires_grad=True)
        hx = torch.zeros(1, 1, 64)

        actor_logits, critic_value, _ = model(x, hx)
        loss = actor_logits.sum() + critic_value.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)
