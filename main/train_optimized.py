"""
Train Optimized - Script di addestramento OTTIMIZZATO per Scopa AI

Ottimizzazioni implementate:
    - SubprocVecEnv: Ambienti paralleli su processi separati (multiprocessing)
    - CPU + GPU: CPU esegue ambienti, GPU esegue forward/backward pass
    - Hyperparameters ottimizzati per VecEnv
    - Benchmark integrato per misurare steps/s

Usage:
    python train_optimized.py --gpu --n-envs 8  # Training con 8 ambienti paralleli
    python train_optimized.py --benchmark       # Benchmark CPU vs GPU
"""
import os
import argparse
import time
import multiprocessing as mp
from datetime import datetime
from functools import partial
from typing import Callable

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from scopa_env import ScopaEnv


# Directory per i log e modelli
LOG_DIR = "./logs/"
MODEL_DIR = "./models/"


def mask_fn(env: ScopaEnv) -> np.ndarray:
    """Funzione per estrarre action masks dall'ambiente."""
    return env.action_masks()


def make_env(opponent_mode: str, rank: int = 0, log_dir: str = None) -> Callable:
    """
    Factory function per creare ambienti compatibili con SubprocVecEnv.
    
    Args:
        opponent_mode: Modalità avversario ('random', 'self', 'heuristic', 'mixed')
        rank: ID univoco dell'ambiente (per seed e log)
        log_dir: Directory per i log Monitor
    
    Returns:
        Funzione che crea l'ambiente
    """
    def _init() -> ScopaEnv:
        env = ScopaEnv(opponent_mode=opponent_mode)
        env.reset(seed=rank)  # Seed diverso per ogni ambiente
        
        # Wrap con ActionMasker per MaskablePPO
        env = ActionMasker(env, mask_fn)
        
        # Monitor per logging (solo il primo ambiente per evitare conflitti)
        if log_dir is not None and rank == 0:
            env = Monitor(env, log_dir, info_keywords=())
        
        return env
    
    return _init


def make_vec_env(
    opponent_mode: str,
    n_envs: int = 8,
    use_subproc: bool = True,
    log_dir: str = None
) -> SubprocVecEnv | DummyVecEnv:
    """
    Crea un ambiente vettorizzato con N copie parallele.
    
    Args:
        opponent_mode: Modalità avversario
        n_envs: Numero di ambienti paralleli
        use_subproc: Se True usa SubprocVecEnv (multiprocessing), altrimenti DummyVecEnv
        log_dir: Directory per i log
    
    Returns:
        Ambiente vettorizzato
    """
    env_fns = [make_env(opponent_mode, rank=i, log_dir=log_dir) for i in range(n_envs)]
    
    if use_subproc and n_envs > 1:
        # SubprocVecEnv: ogni ambiente gira su un processo separato
        # Usa 'forkserver' per compatibilità con CUDA
        return SubprocVecEnv(env_fns, start_method='forkserver')
    else:
        # DummyVecEnv: ambienti sequenziali (per debug o n_envs=1)
        return DummyVecEnv(env_fns)


class SelfPlayCallback(BaseCallback):
    """
    Callback per aggiornare il modello usato per self-play in VecEnv.
    
    NOTA: Con SubprocVecEnv gli ambienti girano in processi separati,
    quindi dobbiamo usare env_method() per comunicare con i worker.
    """
    def __init__(self, update_freq: int = 200000, verbose: int = 0):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.last_update = 0  # Traccia l'ultimo aggiornamento reale
    
    def _on_step(self) -> bool:
        # Usa num_timesteps (totale globale) invece di n_calls
        if self.num_timesteps >= self.last_update + self.update_freq:
            # Sincronizza il modello in TUTTI i processi paralleli
            # env_method() funziona sia con SubprocVecEnv che DummyVecEnv
            self.training_env.env_method("set_model", self.model)
            self.last_update = self.num_timesteps
            
            if self.verbose > 0:
                print(f"[SelfPlay] Modello aggiornato a step {self.num_timesteps:,}")
        return True



class BenchmarkCallback(BaseCallback):
    """
    Callback per misurare steps/s durante il training.
    """
    def __init__(self, log_interval: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.start_time = None
        self.start_timesteps = 0
    
    def _on_training_start(self) -> None:
        self.start_time = time.perf_counter()
        self.start_timesteps = self.num_timesteps
    
    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            elapsed = time.perf_counter() - self.start_time
            timesteps = self.num_timesteps - self.start_timesteps
            steps_per_sec = timesteps / elapsed if elapsed > 0 else 0
            
            if self.verbose > 0:
                n_envs = len(self.training_env.envs) if hasattr(self.training_env, 'envs') else 1
                print(f"📊 Steps: {self.num_timesteps:,} | "
                      f"Speed: {steps_per_sec:.1f} steps/s | "
                      f"Envs: {n_envs}")
        return True


def train(
    total_timesteps: int = 1_000_000,
    opponent_mode: str = 'random',
    continue_from: str | None = None,
    use_gpu: bool = False,
    n_envs: int = 8,
    benchmark: bool = False,
    save_model: bool = True,
    quiet: bool = False
):
    """
    Addestra il modello MaskablePPO con ambienti paralleli.
    
    Args:
        total_timesteps: Numero totale di step
        opponent_mode: Modalità avversario
        continue_from: Path modello esistente
        use_gpu: Se True usa CUDA
        n_envs: Numero di ambienti paralleli
        benchmark: Se True, stampa statistiche di velocità
    """
    device = "cuda" if use_gpu else "cpu"
    
    if not quiet:
        print(f"{'='*60}")
        print(f"🚀 SCOPA AI - Training Ottimizzato")
        print(f"{'='*60}")
        print(f"🖥️  Device: {device.upper()}")
        print(f"🔄 Ambienti paralleli: {n_envs}")
        print(f"🎲 Modalità: {opponent_mode.upper()}")
        print(f"{'='*60}")
    
    # Crea directory
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Crea ambiente vettorizzato
    vec_env = make_vec_env(
        opponent_mode=opponent_mode,
        n_envs=n_envs,
        use_subproc=n_envs > 1,
        log_dir=LOG_DIR
    )
    
    # Hyperparameters ottimizzati per VecEnv
    # n_steps più piccoli perché abbiamo n_envs ambienti
    # batch_size più grande per sfruttare GPU
    n_steps_per_env = max(256, 2048 // n_envs)
    batch_size = min(512, n_steps_per_env * n_envs // 4)
    
    # Carica o crea modello
    if continue_from:
        print(f"📂 Caricamento modello da: {continue_from}")
        model = MaskablePPO.load(
            continue_from,
            env=vec_env,
            ent_coef=0.07,
            learning_rate=0.0003,
            device=device
        )
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=0 if quiet else 1,
            learning_rate=0.0003,
            gamma=0.99,
            n_steps=n_steps_per_env,
            batch_size=batch_size,
            n_epochs=10,
            ent_coef=0.05,
            device=device,
        )
    
    if not quiet:
        print(f"\n📐 Hyperparameters ottimizzati:")
        print(f"   n_steps per env: {n_steps_per_env}")
        print(f"   batch_size: {batch_size}")
        print(f"   total n_steps: {n_steps_per_env * n_envs}")
    
    # Setup callbacks
    callbacks = []
    
    if benchmark:
        callbacks.append(BenchmarkCallback(log_interval=5000, verbose=1))
    
    if opponent_mode in ['self', 'mixed']:
        callbacks.append(SelfPlayCallback(update_freq=200000, verbose=1))
    
    # Training
    if not quiet:
        print(f"\n🎯 Inizio addestramento per {total_timesteps:,} timestep...")
    start_time = time.perf_counter()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None
        )
    finally:
        # Chiudi ambiente vettorizzato
        vec_env.close()
    
    elapsed = time.perf_counter() - start_time
    steps_per_sec = total_timesteps / elapsed
    
    # Salva modello (solo se richiesto)
    model_path = None
    if save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODEL_DIR, f"scopa_ai_{opponent_mode}_{timestamp}")
        model.save(model_path)
        model.save(os.path.join(MODEL_DIR, f"scopa_ai_{opponent_mode}_latest"))
        model.save(os.path.join(MODEL_DIR, "scopa_ai_latest"))
    
    if not quiet:
        print(f"\n{'='*60}")
        print(f"✅ Addestramento completato!")
        print(f"{'='*60}")
        print(f"⏱️  Tempo totale: {elapsed:.1f}s")
        print(f"🚀 Velocità media: {steps_per_sec:.1f} steps/s")
        if model_path:
            print(f"💾 Modello salvato: {model_path}.zip")
        print(f"{'='*60}")
    
    return steps_per_sec


def detect_system_info() -> dict:
    """
    Rileva le caratteristiche del sistema per ottimizzare il benchmark.
    """
    import platform
    import torch
    
    info = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'cpu_count': os.cpu_count() or 4,
        'is_arm': platform.machine().lower() in ('aarch64', 'arm64', 'armv8'),
        'has_gpu': False,
        'gpu_name': None,
    }
    
    # Controlla GPU
    try:
        if torch.cuda.is_available():
            info['has_gpu'] = True
            info['gpu_name'] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    
    return info


def run_benchmark(timesteps: int = 30_000):
    """
    Esegue benchmark automatico per trovare la configurazione ottimale.
    
    - Auto-rileva CPU, GPU, ARM
    - Testa configurazioni sensate per il sistema
    - Trova la configurazione più veloce
    - Non salva modelli durante il benchmark
    """
    print(f"\n{'='*70}")
    print(f"📊 BENCHMARK AUTOMATICO - Trova la configurazione ottimale")
    print(f"{'='*70}")
    
    # Rileva sistema
    sys_info = detect_system_info()
    
    print(f"\n🖥️  Sistema rilevato:")
    print(f"   Platform: {sys_info['platform']} ({sys_info['machine']})")
    print(f"   CPU cores: {sys_info['cpu_count']}")
    print(f"   ARM: {'Sì' if sys_info['is_arm'] else 'No'}")
    print(f"   GPU: {sys_info['gpu_name'] if sys_info['has_gpu'] else 'Non disponibile'}")
    
    # Determina le configurazioni da testare
    configs_to_test = []
    
    # Baseline sempre
    configs_to_test.append(('1 env CPU', {'n_envs': 1, 'use_gpu': False}))
    
    # Configurazioni CPU multiprocessing
    # Testa 2, 4, 8 e eventualmente 12/16 in base ai core
    cpu_count = sys_info['cpu_count']
    env_counts = [2, 4]
    if cpu_count >= 6:
        env_counts.append(8)
    if cpu_count >= 12:
        env_counts.append(12)
    if cpu_count >= 16:
        env_counts.append(16)
    
    for n in env_counts:
        configs_to_test.append((f'{n} env CPU', {'n_envs': n, 'use_gpu': False}))
    
    # Configurazioni GPU se disponibile
    if sys_info['has_gpu']:
        configs_to_test.append(('1 env GPU', {'n_envs': 1, 'use_gpu': True}))
        for n in env_counts:
            configs_to_test.append((f'{n} env GPU', {'n_envs': n, 'use_gpu': True}))
    
    print(f"\n🧪 Configurazioni da testare: {len(configs_to_test)}")
    print(f"   Timesteps per test: {timesteps:,}")
    print(f"{'='*70}\n")
    
    # Esegui benchmark
    results = {}
    total_tests = len(configs_to_test)
    
    for i, (name, config) in enumerate(configs_to_test, 1):
        print(f"\n🔄 Test {i}/{total_tests}: {name}...")
        try:
            speed = train(
                total_timesteps=timesteps,
                use_gpu=config['use_gpu'],
                n_envs=config['n_envs'],
                benchmark=False,  # Disabilita log durante benchmark
                save_model=False,  # NON salvare modello
                quiet=True  # Modalità silenziosa
            )
            results[name] = speed
            print(f"   ✅ {speed:,.1f} steps/s")
        except Exception as e:
            print(f"   ❌ Errore: {e}")
            results[name] = 0
    
    # Trova la configurazione migliore
    valid_results = {k: v for k, v in results.items() if v > 0}
    if not valid_results:
        print("\n❌ Nessun test completato con successo!")
        return
    
    best_config = max(valid_results, key=valid_results.get)
    best_speed = valid_results[best_config]
    baseline_speed = results.get('1 env CPU', 1)
    
    # Risultati ordinati
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*70}")
    print(f"📊 RISULTATI BENCHMARK")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Configurazione':<25} {'Steps/s':>12} {'Speedup':>10}")
    print(f"{'-'*70}")
    
    for rank, (config, speed) in enumerate(sorted_results, 1):
        if speed > 0:
            speedup = speed / baseline_speed if baseline_speed > 0 else 0
            marker = " 👑" if config == best_config else ""
            print(f"{rank:<6} {config:<25} {speed:>12,.1f} {speedup:>9.2f}x{marker}")
        else:
            print(f"{rank:<6} {config:<25} {'FALLITO':>12} {'-':>10}")
    
    print(f"{'='*70}")
    
    # Suggerimento finale
    print(f"\n🏆 CONFIGURAZIONE OTTIMALE: {best_config}")
    print(f"   Velocità: {best_speed:,.1f} steps/s")
    if baseline_speed > 0:
        print(f"   Speedup: {best_speed / baseline_speed:.2f}x rispetto a baseline")
    
    # Genera comando suggerito
    best_cfg = dict(configs_to_test)[best_config]
    cmd_parts = ["python train_optimized.py"]
    if best_cfg['use_gpu']:
        cmd_parts.append("--gpu")
    cmd_parts.append(f"--n-envs {best_cfg['n_envs']}")
    
    print(f"\n📋 Comando consigliato:")
    print(f"   {' '.join(cmd_parts)}")
    print(f"{'='*70}")
    
    return results


if __name__ == "__main__":
    # Imposta metodo di avvio multiprocessing per compatibilità CUDA
    mp.set_start_method('forkserver', force=True)
    
    parser = argparse.ArgumentParser(
        description='Scopa AI - Training Ottimizzato',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
    python train_optimized.py --gpu --n-envs 8       # Training standard
    python train_optimized.py --benchmark            # Benchmark configurazioni
    python train_optimized.py --gpu --n-envs 16 -t 5000000  # Training intensivo
        """
    )
    
    parser.add_argument('--mode', '-m', type=str, default='random',
                        choices=['random', 'self', 'heuristic', 'mixed'],
                        help='Modalità avversario (default: random)')
    parser.add_argument('--timesteps', '-t', type=int, default=1_000_000,
                        help='Numero totale di timesteps (default: 1000000)')
    parser.add_argument('--continue-from', '-c', type=str,
                        default='./models/scopa_ai_latest',
                        help='Path modello da cui continuare')
    parser.add_argument('--fresh', '-f', action='store_true',
                        help='Ignora modello esistente')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Usa GPU (CUDA)')
    parser.add_argument('--n-envs', '-n', type=int, default=8,
                        help='Numero di ambienti paralleli (default: 8)')
    parser.add_argument('--benchmark', '-b', action='store_true',
                        help='Esegui benchmark comparativo')
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmark(timesteps=30_000)
    else:
        # Gestione modello esistente
        continue_from = args.continue_from
        if args.fresh:
            continue_from = None
            print("🆕 Avvio fresh (ignoro modelli esistenti)")
        elif continue_from and not os.path.exists(continue_from + '.zip'):
            print(f"⚠️  Modello non trovato: {continue_from}.zip")
            print("   Avvio nuovo addestramento da zero...")
            continue_from = None
        
        train(
            total_timesteps=args.timesteps,
            opponent_mode=args.mode,
            continue_from=continue_from,
            use_gpu=args.gpu,
            n_envs=args.n_envs,
            benchmark=True  # Sempre mostra velocità
        )
