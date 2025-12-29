"""
Scopa Training Module - Training utilities.

Per il training attivo, usare:
    python scripts/train_recurrent.py

Exports:
- RecurrentRolloutBuffer: Buffer per policy ricorrenti
- Callbacks: SelfPlayCallback, BenchmarkCallback, ValueLossLoggerCallback
- plot_results: Visualizzazione risultati training
"""
from scopa.training.callbacks import SelfPlayCallback, BenchmarkCallback, ValueLossLoggerCallback
from scopa.training.rollout_buffer import RecurrentRolloutBuffer
from scopa.training.visualization import plot_results

__all__ = [
    "RecurrentRolloutBuffer",
    "SelfPlayCallback",
    "BenchmarkCallback",
    "ValueLossLoggerCallback",
    "plot_results",
]
