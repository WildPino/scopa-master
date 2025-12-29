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


class ValueLossLoggerCallback(BaseCallback):
    """
    Callback per loggare la value_loss in un CSV.
    
    Questo permette alla visualization di mostrare la convergenza
    della value_loss (il "logaritmo rovesciato" che indica un buon training).
    
    Args:
        log_dir: Directory dove salvare il CSV
        log_interval: Intervallo tra i log (in n_updates)
        verbose: Livello di verbosità
    """
    
    def __init__(self, log_dir: str, log_interval: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.csv_path = None
        self.last_n_updates = 0
    
    def _on_training_start(self) -> None:
        from pathlib import Path
        self.csv_path = Path(self.log_dir) / "value_loss.csv"
        # Scrivi header se file non esiste
        if not self.csv_path.exists():
            self.csv_path.write_text("timesteps,n_updates,value_loss\n")
    
    def _on_step(self) -> bool:
        # Leggi value_loss dal logger del modello
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            # Ottieni n_updates corrente
            n_updates = getattr(self.model, '_n_updates', 0)
            
            # Logga solo se ci sono stati nuovi update
            if n_updates > self.last_n_updates and n_updates % self.log_interval == 0:
                # Prova a leggere value_loss dai name_to_value del logger
                logger = self.model.logger
                value_loss = None
                
                if hasattr(logger, 'name_to_value'):
                    value_loss = logger.name_to_value.get('train/value_loss')
                
                if value_loss is not None:
                    with open(self.csv_path, 'a') as f:
                        f.write(f"{self.num_timesteps},{n_updates},{value_loss}\n")
                    
                    if self.verbose > 0:
                        print(f"[ValueLoss] Step {self.num_timesteps:,}: {value_loss:.4f}")
                
                self.last_n_updates = n_updates
        
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
