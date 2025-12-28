"""
Training Callbacks - Callbacks personalizzate per il training.

Contiene:
- SelfPlayCallback: Aggiorna il modello per self-play
- BenchmarkCallback: Misura steps/s durante training
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv


class SelfPlayCallback(BaseCallback):
    """
    Callback per aggiornare i pesi della policy per self-play in VecEnv.
    
    Con SubprocVecEnv gli ambienti girano in processi separati.
    Non possiamo inviare l'intero modello (non serializzabile), quindi
    inviamo solo lo state_dict della policy (tensori serializzabili).
    
    Args:
        update_freq: Frequenza di aggiornamento in timesteps
        verbose: Livello di verbosità
    """
    
    def __init__(self, update_freq: int = 200_000, verbose: int = 0):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.last_update = 0
    
    def _on_step(self) -> bool:
        if self.num_timesteps >= self.last_update + self.update_freq:
            # Estrai lo state_dict della policy (serializzabile)
            state_dict = self.model.policy.state_dict()
            # Converti tensori a CPU per serializzazione
            state_dict_cpu = {k: v.cpu() for k, v in state_dict.items()}
            
            self.training_env.env_method("update_model_weights", state_dict_cpu)
            self.last_update = self.num_timesteps
            
            if self.verbose > 0:
                print(f"[SelfPlay] Pesi modello aggiornati a step {self.num_timesteps:,}")
        
        return True


class BenchmarkCallback(BaseCallback):
    """
    Callback per misurare steps/s durante il training.
    
    Args:
        log_interval: Intervallo di logging in steps
        verbose: Livello di verbosità
    """
    
    def __init__(self, log_interval: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.start_time: float = 0.0
        self.start_timesteps: int = 0
    
    def _on_training_start(self) -> None:
        self.start_time = time.perf_counter()
        self.start_timesteps = self.num_timesteps
    
    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            elapsed = time.perf_counter() - self.start_time
            timesteps = self.num_timesteps - self.start_timesteps
            steps_per_sec = timesteps / elapsed if elapsed > 0 else 0
            
            if self.verbose > 0:
                n_envs = getattr(self.training_env, 'num_envs', 1)
                print(
                    f"📊 Steps: {self.num_timesteps:,} | "
                    f"Speed: {steps_per_sec:.1f} steps/s | "
                    f"Envs: {n_envs}"
                )
        
        return True
