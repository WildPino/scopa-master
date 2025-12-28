"""
Scopa RL Module - Reinforcement Learning components.

Exports:
- ScopaEnv: Gymnasium-compatible environment
- ScopaNet: Custom neural network (experimental)
"""
from scopa.rl.environment import ScopaEnv
from scopa.rl.networks import ScopaNet

__all__ = ["ScopaEnv", "ScopaNet"]
