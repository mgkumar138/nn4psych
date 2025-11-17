"""
NN4Psych: Neural Networks for Psychological Modeling

A modular package for training and analyzing RNN actor-critic models
on predictive inference tasks.
"""

__version__ = "0.2.0"

from nn4psych.models.actor_critic import ActorCritic
from nn4psych.envs.predictive_inference import PIE_CP_OB_v2

__all__ = [
    "ActorCritic",
    "PIE_CP_OB_v2",
    "__version__",
]
