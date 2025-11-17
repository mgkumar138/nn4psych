# Archived Legacy Code

This directory contains legacy implementations that have been superseded by the new modular `nn4psych` package.

## Contents

### v0/ - Early JAX Implementations (Outdated)
- `context_bandits.py` - JAX-based actor-critic for context bandits
- `rnn_helicopter.py` - JAX RNN implementation
- `play_helicopter_discrete.py` - Pygame visualization

### v1/ - JAX Actor-Critic Variants
- `main.py` - Two-arm bandit without context
- `main_healthy.py` - Change-point task
- `main_healthy_ff.py` - Feed-forward variant

### toy/ - Simplified Prototype
- `model.py` - Reservoir-style actor-critic (NumPy)
- `main.py` - Minimal training loop
- `task.py` - Simplified task environment

## Notes

- These implementations use JAX or simplified NumPy approaches
- They are **not maintained** and may have compatibility issues
- Use the canonical implementations in `nn4psych/` instead
- Kept for historical reference and potential migration

## When to Use

- Reference for understanding evolution of the codebase
- Comparison with JAX implementations
- Educational purposes

## Not Recommended For

- New experiments
- Production use
- Active development
