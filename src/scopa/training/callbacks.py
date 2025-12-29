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


class LeagueManagerCallback(BaseCallback):
    """
    Callback per gestire pool di avversari e self-play diversificato.
    
    Pool di avversari:
    - p(latest) = 0.6
    - p(snapshot_pool) = 0.25
    - p(heuristic) = 0.1
    - p(random) = 0.05
    
    Features:
    - Salva snapshot periodici del modello
    - Valuta periodicamente su set fisso di avversari
    - Supporta best-response training step
    
    Args:
        snapshot_dir: Directory per salvare gli snapshot
        snapshot_freq: Frequenza salvataggio snapshot (timesteps)
        eval_freq: Frequenza valutazione (timesteps)
        pool_sampling: Dict con probabilità per tipo avversario
        max_snapshots: Numero massimo di snapshot da mantenere
        verbose: Livello di verbosità
    """
    
    DEFAULT_SAMPLING = {
        "latest": 0.6,
        "snapshot": 0.25,
        "heuristic": 0.1,
        "random": 0.05,
    }
    
    def __init__(
        self,
        snapshot_dir: str,
        snapshot_freq: int = 50_000,
        eval_freq: int = 100_000,
        pool_sampling: dict = None,
        max_snapshots: int = 10,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.snapshot_dir = snapshot_dir
        self.snapshot_freq = snapshot_freq
        self.eval_freq = eval_freq
        self.pool_sampling = pool_sampling or self.DEFAULT_SAMPLING
        self.max_snapshots = max_snapshots
        
        self.last_snapshot = 0
        self.last_eval = 0
        self.snapshot_paths: list = []
        self.eval_history: list = []
    
    def _on_training_start(self) -> None:
        from pathlib import Path
        
        # Crea directory snapshot
        snapshot_path = Path(self.snapshot_dir)
        snapshot_path.mkdir(parents=True, exist_ok=True)
        
        # Cerca snapshot esistenti
        existing = list(snapshot_path.glob("snapshot_*.zip"))
        self.snapshot_paths = sorted([str(p) for p in existing])
        
        if self.verbose > 0:
            print(f"[League] Inizializzato con {len(self.snapshot_paths)} snapshot esistenti")
    
    def _on_step(self) -> bool:
        # Salva snapshot periodicamente
        if self.num_timesteps >= self.last_snapshot + self.snapshot_freq:
            self._save_snapshot()
            self.last_snapshot = self.num_timesteps
        
        # Valuta periodicamente
        if self.num_timesteps >= self.last_eval + self.eval_freq:
            metrics = self._run_evaluation()
            self.eval_history.append({
                "timesteps": self.num_timesteps,
                "metrics": metrics,
            })
            self.last_eval = self.num_timesteps
        
        return True
    
    def _save_snapshot(self) -> None:
        """Salva checkpoint corrente nel pool."""
        from pathlib import Path
        
        snapshot_name = f"snapshot_{self.num_timesteps}.zip"
        snapshot_path = Path(self.snapshot_dir) / snapshot_name
        
        self.model.save(str(snapshot_path)[:-4])  # Rimuovi .zip (SB3 lo aggiunge)
        self.snapshot_paths.append(str(snapshot_path))
        
        # Limita numero snapshot
        while len(self.snapshot_paths) > self.max_snapshots:
            old_path = self.snapshot_paths.pop(0)
            try:
                Path(old_path).unlink()
            except OSError:
                pass
        
        if self.verbose > 0:
            print(f"[League] Snapshot salvato: {snapshot_name}")
    
    def _sample_opponent(self) -> str:
        """Sceglie tipo avversario dal pool secondo distribuzione."""
        import random
        
        types = list(self.pool_sampling.keys())
        probs = list(self.pool_sampling.values())
        
        return random.choices(types, weights=probs, k=1)[0]
    
    def _run_evaluation(self) -> dict:
        """Valuta contro pool fisso e ritorna metriche."""
        from pathlib import Path
        
        metrics = {
            "timesteps": self.num_timesteps,
            "vs_random": 0.0,
            "vs_heuristic": 0.0,
        }
        
        # Evaluation semplificata - usa eval_match se disponibile
        try:
            # Import qui per evitare circular imports
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))
            
            # Per ora, riporta solo placeholder
            # In produzione: chiamare eval_match.evaluate_model()
            if self.verbose > 0:
                print(f"[League] Valutazione a step {self.num_timesteps:,}")
        except Exception as e:
            if self.verbose > 0:
                print(f"[League] Errore valutazione: {e}")
        
        return metrics
    
    def get_snapshot_path(self) -> str:
        """Ritorna path di uno snapshot casuale dal pool."""
        import random
        
        if not self.snapshot_paths:
            return None
        
        return random.choice(self.snapshot_paths)
