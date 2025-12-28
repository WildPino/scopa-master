"""
Scopa Training Module - Training scripts and utilities.

Exports:
- train (function): Basic training
- train_parallel (function): Optimized parallel training
- plot_results (function): Training visualization
"""
from scopa.training.callbacks import SelfPlayCallback, BenchmarkCallback

__all__ = ["SelfPlayCallback", "BenchmarkCallback"]
