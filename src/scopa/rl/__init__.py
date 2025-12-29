"""
Scopa RL Module - Reinforcement Learning components.

Exports:
- ScopaEnv: Gymnasium-compatible environment
- MaskableRecurrentPolicy: Policy ricorrente con action masking (ATTIVO)
- BeliefNet: Network per stima carte avversario

Legacy (per compatibilità):
- ScopaNet: Rete MLP base (usata da vecchi modelli MaskablePPO)
"""
from scopa.rl.environment import ScopaEnv
from scopa.rl.policies import MaskableRecurrentPolicy
from scopa.rl.networks import ScopaNet
from scopa.rl.belief import BeliefNet, BeliefTrainer, BeliefMetrics

__all__ = [
    # Core
    "ScopaEnv",
    "MaskableRecurrentPolicy",
    # Belief
    "BeliefNet",
    "BeliefTrainer",
    "BeliefMetrics",
    # Legacy
    "ScopaNet",
]
