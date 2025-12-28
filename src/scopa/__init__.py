"""
Scopa AI - Reinforcement Learning agent for the Italian card game Scopa.

This package provides:
- Game engine for Scopa rules
- Gymnasium-compatible RL environment
- Training scripts with MaskablePPO
- Pygame GUI for playing against the AI
"""
__version__ = "1.0.0"

from scopa.game import Card, Suit, ScopaEngine
from scopa.rl import ScopaEnv

__all__ = ["Card", "Suit", "ScopaEngine", "ScopaEnv", "__version__"]
