"""
Train Parallel - Script di addestramento OTTIMIZZATO con SubprocVecEnv.

Ottimizzazioni:
- SubprocVecEnv: Ambienti paralleli su processi separati
- CPU + GPU: CPU esegue ambienti, GPU esegue forward/backward pass
- Hyperparameters ottimizzati per VecEnv
- Benchmark integrato per misurare steps/s

Usage:
    python -m scopa.training.train_parallel --gpu --n-envs 8
    python -m scopa.training.train_parallel --benchmark
"""
from __future__ import annotations

import argparse
import os
import platform
import time
from datetime import datetime
from typing import Callable, Optional

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from scopa.rl import ScopaEnv
from scopa.config import (
    MODELS_DIR, LOGS_DIR,
    DEFAULT_HYPERPARAMS, NETWORK_ARCH, OPPONENT_MODES,
    ensure_dirs
)
from scopa.training.callbacks import SelfPlayCallback, BenchmarkCallback


def linear_schedule(
    initial_value: float,
    final_value: float = 0.00005
) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    
    Riduce il LR linearmente da initial_value a final_value.
    """
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func


def mask_fn(env: ScopaEnv) -> np.ndarray:
    """Estrae action masks dall'ambiente."""
    return env.action_masks()


def make_env(
    opponent_mode: str,
    rank: int = 0,
    log_dir: Optional[str] = None
) -> Callable[[], ScopaEnv]:
    """Factory function per creare ambienti compatibili con SubprocVecEnv."""
    def _init() -> ScopaEnv:
        env = ScopaEnv(opponent_mode=opponent_mode)
        env.reset(seed=rank)
        env = ActionMasker(env, mask_fn)
        
        if log_dir is not None and rank == 0:
            env = Monitor(env, log_dir)
        
        return env
    
    return _init


def make_vec_env(
    opponent_mode: str,
    n_envs: int = 8,
    use_subproc: bool = True,
    log_dir: Optional[str] = None
) -> SubprocVecEnv | DummyVecEnv:
    """Crea ambiente vettorizzato con N copie parallele."""
    env_fns = [make_env(opponent_mode, rank=i, log_dir=log_dir) for i in range(n_envs)]
    
    if use_subproc and n_envs > 1:
        start_method = "spawn" if platform.system() == "Windows" else "forkserver"
        return SubprocVecEnv(env_fns, start_method=start_method)
    else:
        return DummyVecEnv(env_fns)


def train(
    total_timesteps: int = 1_000_000,
    opponent_mode: str = "random",
    continue_from: Optional[str] = None,
    use_gpu: bool = False,
    n_envs: int = 8,
    benchmark: bool = False,
    save_model: bool = True,
    quiet: bool = False
) -> float:
    """
    Addestra il modello MaskablePPO con ambienti paralleli.
    
    Returns:
        steps/s velocità raggiunta
    """
    device = "cuda" if use_gpu else "cpu"
    
    if not quiet:
        print("=" * 60)
        print("🚀 SCOPA AI - Training Ottimizzato")
        print("=" * 60)
        print(f"🖥️  Device: {device.upper()}")
        print(f"🔄 Ambienti paralleli: {n_envs}")
        print(f"🎲 Modalità: {opponent_mode.upper()}")
        print("=" * 60)
    
    ensure_dirs()
    
    # Crea ambiente vettorizzato
    vec_env = make_vec_env(
        opponent_mode=opponent_mode,
        n_envs=n_envs,
        use_subproc=n_envs > 1,
        log_dir=str(LOGS_DIR)
    )
    
    # Hyperparameters ottimizzati
    hp = DEFAULT_HYPERPARAMS
    n_steps_per_env = max(256, 2048 // n_envs)
    batch_size = 1024
    
    policy_kwargs = dict(net_arch=dict(pi=NETWORK_ARCH["pi"], vf=NETWORK_ARCH["vf"]))
    
    # Carica o crea modello
    if continue_from:
        print(f"📂 Caricamento modello da: {continue_from}")
        model = MaskablePPO.load(
            continue_from,
            env=vec_env,
            ent_coef=hp["ent_coef"],
            learning_rate=hp["learning_rate"],
            device=device
        )
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=0 if quiet else 1,
            learning_rate=linear_schedule(hp["learning_rate"], hp["learning_rate_final"]),
            gamma=hp["gamma"],
            n_steps=n_steps_per_env,
            batch_size=batch_size,
            n_epochs=hp["n_epochs"],
            ent_coef=hp["ent_coef"],
            policy_kwargs=policy_kwargs,
            device=device,
        )
    
    if not quiet:
        print(f"\n📐 Hyperparameters:")
        print(f"   n_steps per env: {n_steps_per_env}")
        print(f"   batch_size: {batch_size}")
        print(f"   total n_steps: {n_steps_per_env * n_envs}")
    
    # Setup callbacks
    callbacks = []
    if benchmark:
        callbacks.append(BenchmarkCallback(log_interval=5000, verbose=1))
    if opponent_mode in ["self", "mixed"]:
        callbacks.append(SelfPlayCallback(update_freq=200_000, verbose=1))
    
    # Training
    if not quiet:
        print(f"\n🎯 Inizio addestramento per {total_timesteps:,} timestep...")
    
    start_time = time.perf_counter()
    
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks or None)
    finally:
        vec_env.close()
    
    elapsed = time.perf_counter() - start_time
    steps_per_sec = total_timesteps / elapsed
    
    # Salva modello
    model_path = None
    if save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = MODELS_DIR / f"scopa_ai_{opponent_mode}_{timestamp}"
        model.save(str(model_path))
        model.save(str(MODELS_DIR / f"scopa_ai_{opponent_mode}_latest"))
        model.save(str(MODELS_DIR / "scopa_ai_latest"))
    
    if not quiet:
        print("\n" + "=" * 60)
        print("✅ Addestramento completato!")
        print("=" * 60)
        print(f"⏱️  Tempo totale: {elapsed:.1f}s")
        print(f"🚀 Velocità media: {steps_per_sec:.1f} steps/s")
        if model_path:
            print(f"💾 Modello salvato: {model_path}.zip")
        print("=" * 60)
    
    return steps_per_sec


def run_benchmark(timesteps: int = 30_000) -> dict:
    """Esegue benchmark automatico per trovare la configurazione ottimale."""
    import torch
    
    print("\n" + "=" * 70)
    print("📊 BENCHMARK AUTOMATICO")
    print("=" * 70)
    
    # Info sistema
    cpu_count = os.cpu_count() or 4
    has_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else None
    
    print(f"\n🖥️  Sistema:")
    print(f"   CPU cores: {cpu_count}")
    print(f"   GPU: {gpu_name if has_gpu else 'Non disponibile'}")
    
    # Configurazioni da testare
    configs = [("1 env CPU", {"n_envs": 1, "use_gpu": False})]
    
    env_counts = [2, 4]
    if cpu_count >= 6:
        env_counts.append(8)
    if cpu_count >= 12:
        env_counts.append(12)
    
    for n in env_counts:
        configs.append((f"{n} env CPU", {"n_envs": n, "use_gpu": False}))
    
    if has_gpu:
        configs.append(("1 env GPU", {"n_envs": 1, "use_gpu": True}))
        for n in env_counts:
            configs.append((f"{n} env GPU", {"n_envs": n, "use_gpu": True}))
    
    print(f"\n🧪 Configurazioni da testare: {len(configs)}")
    print(f"   Timesteps per test: {timesteps:,}\n")
    
    # Esegui benchmark
    results = {}
    for i, (name, cfg) in enumerate(configs, 1):
        print(f"🔄 Test {i}/{len(configs)}: {name}...")
        try:
            speed = train(
                total_timesteps=timesteps,
                use_gpu=cfg["use_gpu"],
                n_envs=cfg["n_envs"],
                benchmark=False,
                save_model=False,
                quiet=True
            )
            results[name] = speed
            print(f"   ✅ {speed:,.1f} steps/s")
        except Exception as e:
            print(f"   ❌ Errore: {e}")
            results[name] = 0
    
    # Risultati
    valid = {k: v for k, v in results.items() if v > 0}
    if not valid:
        print("\n❌ Nessun test completato!")
        return results
    
    best = max(valid, key=valid.get)
    baseline = results.get("1 env CPU", 1)
    
    print("\n" + "=" * 70)
    print("📊 RISULTATI")
    print("=" * 70)
    
    for rank, (cfg, speed) in enumerate(sorted(results.items(), key=lambda x: -x[1]), 1):
        if speed > 0:
            marker = " 👑" if cfg == best else ""
            print(f"{rank}. {cfg:20} {speed:>10,.1f} steps/s  {speed/baseline:.2f}x{marker}")
    
    print("\n" + "=" * 70)
    print(f"🏆 MIGLIORE: {best} ({valid[best]:,.1f} steps/s)")
    print("=" * 70)
    
    # Genera comando ideale
    best_cfg = next((cfg for name, cfg in configs if name == best), None)
    if best_cfg:
        gpu_flag = "--gpu " if best_cfg["use_gpu"] else ""
        n_envs = best_cfg["n_envs"]
        print(f"\n💡 COMANDO IDEALE per il tuo sistema:")
        print(f"   python -m scopa.training.train_parallel {gpu_flag}--n-envs {n_envs} --timesteps 10000000 --fresh")
        print("")
    
    return results


def main():
    """Entry point CLI."""
    # Imposta multiprocessing per Windows
    if platform.system() != "Windows":
        import multiprocessing as mp
        mp.set_start_method("forkserver", force=True)
    
    parser = argparse.ArgumentParser(description="Scopa AI - Training Ottimizzato")
    parser.add_argument("--mode", "-m", type=str, default="mixed", choices=OPPONENT_MODES)
    parser.add_argument("--timesteps", "-t", type=int, default=1_000_000)
    parser.add_argument("--continue-from", "-c", type=str, default=str(MODELS_DIR / "scopa_ai_latest"))
    parser.add_argument("--fresh", "-f", action="store_true")
    parser.add_argument("--gpu", "-g", action="store_true")
    parser.add_argument("--n-envs", "-n", type=int, default=8)
    parser.add_argument("--benchmark", "-b", action="store_true")
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmark(timesteps=30_000)
    else:
        continue_from = args.continue_from
        if args.fresh:
            continue_from = None
            print("🆕 Avvio fresh")
        elif continue_from and not os.path.exists(continue_from + ".zip"):
            print(f"⚠️  Modello non trovato: {continue_from}.zip")
            continue_from = None
        
        train(
            total_timesteps=args.timesteps,
            opponent_mode=args.mode,
            continue_from=continue_from,
            use_gpu=args.gpu,
            n_envs=args.n_envs,
            benchmark=True
        )


if __name__ == "__main__":
    main()
