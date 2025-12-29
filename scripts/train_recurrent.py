"""
Train Recurrent - Training loop per MaskableRecurrentPolicy.

Usa config da scopa.config.RECURRENT_TRAINING_CONFIG.
Continua training da checkpoint esistente se non --fresh.

Usage:
    # Training da zero
    python scripts/train_recurrent.py --timesteps 10_000_000 --fresh
    
    # Continua training esistente
    python scripts/train_recurrent.py --timesteps 20_000_000
    
    # Benchmark del modello corrente (SENZA training)
    python scripts/train_recurrent.py --benchmark
    
    # Con GPU
    python scripts/train_recurrent.py --timesteps 10_000_000 --device cuda

Flags:
    --timesteps, -t    Numero totale di timesteps
    --fresh            Inizia da zero (ignora checkpoint esistenti)
    --benchmark        Solo valuta modello corrente (no training)
    --device           cpu o cuda
    --n-envs           Override numero ambienti paralleli
    --opponent         Tipo avversario (random, heuristic, self)
    --eval-freq        Frequenza valutazione in timesteps
    --save-freq        Frequenza salvataggio checkpoint
    --name             Nome del run (sottocartella in models/)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from scopa.rl import ScopaEnv
from scopa.rl.policies import MaskableRecurrentPolicy
from scopa.training.rollout_buffer import RecurrentRolloutBuffer
from scopa.config import (
    MODELS_DIR, LOGS_DIR, ensure_dirs,
    RECURRENT_TRAINING_CONFIG as CONFIG,
    LSTM_CONFIG,
    DEFAULT_MODEL_NAME,
)


def linear_schedule(initial: float, final: float, progress: float) -> float:
    """Linear interpolation."""
    return initial + (final - initial) * progress


def evaluate_policy(
    policy: MaskableRecurrentPolicy,
    opponent: str,
    n_games: int = 100,
) -> Dict[str, float]:
    """Valuta policy contro un opponent."""
    env = ScopaEnv(opponent_mode=opponent)
    
    wins = 0
    total_score_diff = 0
    
    for _ in range(n_games):
        obs, _ = env.reset()
        lstm_states = None
        episode_start = True
        done = False
        
        while not done:
            mask = env.action_masks()
            action, lstm_states = policy.predict(
                obs,
                lstm_states=lstm_states,
                episode_starts=np.array([episode_start]),
                action_masks=mask,
                deterministic=True,
            )
            episode_start = False
            obs, _, done, _, _ = env.step(action)
        
        scores = env.engine.calculate_score()
        if scores[0] > scores[1]:
            wins += 1
        total_score_diff += scores[0] - scores[1]
    
    env.close()
    
    return {
        "winrate": wins / n_games,
        "avg_score_diff": total_score_diff / n_games,
    }


def load_checkpoint(policy: MaskableRecurrentPolicy, save_dir: Path) -> int:
    """
    Carica ultimo checkpoint se esiste.
    
    Returns:
        timesteps già completati (0 se nessun checkpoint)
    """
    # Cerca checkpoints
    checkpoints = list(save_dir.glob("checkpoint_*M.pt"))
    if not checkpoints:
        # Prova con best o final
        if (save_dir / "final.pt").exists():
            policy.load_state_dict(torch.load(save_dir / "final.pt"))
            return 0  # Non sappiamo quanti timesteps
        return 0
    
    # Trova il più recente
    def get_timesteps(p: Path) -> int:
        name = p.stem  # checkpoint_5M
        try:
            return int(name.split("_")[1].replace("M", "")) * 1_000_000
        except:
            return 0
    
    latest = max(checkpoints, key=get_timesteps)
    policy.load_state_dict(torch.load(latest))
    completed = get_timesteps(latest)
    
    print(f"📂 Caricato checkpoint: {latest.name} ({completed:,} steps)")
    return completed


def run_benchmark(
    name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    n_games: int = 100,
) -> None:
    """
    Valuta il modello corrente e mostra la configurazione.
    Non fa training, solo evaluation.
    """
    import gymnasium as gym
    
    ensure_dirs()
    save_dir = MODELS_DIR / name
    
    # Banner
    print(f"\n{'='*60}")
    print("📊 BENCHMARK - Configurazione e Valutazione")
    print(f"{'='*60}")
    
    # Mostra config
    print(f"\n📁 Model directory: {save_dir}")
    print(f"\n⚙️  CONFIGURAZIONE TRAINING (da config.py):")
    print(f"    n_envs: {CONFIG['n_envs']}")
    print(f"    n_steps: {CONFIG['n_steps']}")
    print(f"    batch_size: {CONFIG['batch_size']}")
    print(f"    n_epochs: {CONFIG['n_epochs']}")
    print(f"    lr: {CONFIG['lr_initial']} → {CONFIG['lr_final']}")
    print(f"    ent_coef: {CONFIG['ent_coef_initial']} → {CONFIG['ent_coef_final']}")
    print(f"    gamma: {CONFIG['gamma']}")
    print(f"    gae_lambda: {CONFIG['gae_lambda']}")
    print(f"    clip_range: {CONFIG['clip_range']}")
    print(f"    opponent: {CONFIG['opponent_mode']}")
    
    print(f"\n🧠 LSTM ARCHITECTURE:")
    print(f"    features_dim: {LSTM_CONFIG['features_dim']}")
    print(f"    lstm_hidden_size: {LSTM_CONFIG['lstm_hidden_size']}")
    print(f"    lstm_num_layers: {LSTM_CONFIG['lstm_num_layers']}")
    
    # Cerca checkpoints
    print(f"\n💾 CHECKPOINTS in {save_dir}:")
    if save_dir.exists():
        checkpoints = sorted(save_dir.glob("*.pt"))
        if checkpoints:
            for cp in checkpoints:
                size_mb = cp.stat().st_size / (1024 * 1024)
                print(f"    {cp.name} ({size_mb:.1f} MB)")
        else:
            print("    (nessun checkpoint trovato)")
    else:
        print("    (directory non esiste)")
    
    # Carica e valuta modello
    obs_space = gym.spaces.Box(low=0, high=1, shape=(296,), dtype=np.float32)
    act_space = gym.spaces.Discrete(40)
    
    policy = MaskableRecurrentPolicy(
        obs_space, act_space,
        features_dim=LSTM_CONFIG["features_dim"],
        lstm_hidden_size=LSTM_CONFIG["lstm_hidden_size"],
        lstm_num_layers=LSTM_CONFIG["lstm_num_layers"],
    ).to(device)
    
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"\n🔢 Parametri totali: {n_params:,}")
    
    # Prova a caricare checkpoint
    loaded_steps = load_checkpoint(policy, save_dir) if save_dir.exists() else 0
    
    if loaded_steps == 0 and not any(save_dir.glob("*.pt")) if save_dir.exists() else True:
        print("\n⚠️  Nessun modello allenato trovato. Policy random.")
    
    # Evaluation
    print(f"\n🎯 VALUTAZIONE ({n_games} partite per opponent):")
    
    for opponent in ["random", "heuristic"]:
        metrics = evaluate_policy(policy, opponent, n_games)
        wr = metrics['winrate']
        diff = metrics['avg_score_diff']
        print(f"    vs {opponent:12} | WR: {wr:6.1%} | Δpts: {diff:+.2f}")
    
    print(f"\n{'='*60}")
    print("Per iniziare training:")
    print(f"  python scripts/train_recurrent.py --timesteps 10_000_000 --fresh")
    print("Per continuare training:")
    print(f"  python scripts/train_recurrent.py --timesteps 20_000_000")
    print(f"{'='*60}\n")


def run_hw_benchmark(timesteps: int = 30_000) -> dict:
    """
    Benchmark hardware per trovare la configurazione ottimale.
    
    Testa diverse combinazioni di:
    - CPU vs GPU
    - Numero di ambienti paralleli
    
    Ritorna la configurazione più veloce.
    """
    import os
    import gymnasium as gym
    
    print("\n" + "=" * 70)
    print("🔧 HARDWARE BENCHMARK - Trova configurazione ottimale")
    print("=" * 70)
    
    # Info sistema
    cpu_count = os.cpu_count() or 4
    has_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else None
    
    print(f"\n🖥️  Sistema:")
    print(f"   CPU cores: {cpu_count}")
    print(f"   GPU: {gpu_name if has_gpu else 'Non disponibile'}")
    print(f"   PyTorch: {torch.__version__}")
    
    # Configurazioni da testare
    configs = []
    
    # CPU configurations
    env_counts = [1, 2, 4]
    if cpu_count >= 8:
        env_counts.append(8)
    if cpu_count >= 12:
        env_counts.append(12)
    if cpu_count >= 16:
        env_counts.append(16)
    
    for n in env_counts:
        configs.append((f"{n:2d} env CPU", {"n_envs": n, "device": "cpu"}))
    
    # GPU configurations  
    if has_gpu:
        for n in env_counts:
            configs.append((f"{n:2d} env GPU", {"n_envs": n, "device": "cuda"}))
    
    print(f"\n🧪 Configurazioni da testare: {len(configs)}")
    print(f"   Timesteps per test: {timesteps:,}")
    print(f"   Stima tempo: ~{len(configs) * 30}s\n")
    
    # Esegui benchmark
    obs_space = gym.spaces.Box(low=0, high=1, shape=(296,), dtype=np.float32)
    act_space = gym.spaces.Discrete(40)
    
    results = {}
    
    for i, (name, cfg) in enumerate(configs, 1):
        print(f"🔄 Test {i}/{len(configs)}: {name}...", end=" ", flush=True)
        
        try:
            device = cfg["device"]
            n_envs = cfg["n_envs"]
            n_steps = 256  # Steps per rollout (fisso per benchmark)
            
            # Crea policy
            policy = MaskableRecurrentPolicy(
                obs_space, act_space,
                features_dim=LSTM_CONFIG["features_dim"],
                lstm_hidden_size=LSTM_CONFIG["lstm_hidden_size"],
                lstm_num_layers=LSTM_CONFIG["lstm_num_layers"],
            ).to(device)
            
            optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
            
            # Crea buffer
            buffer = RecurrentRolloutBuffer(
                buffer_size=n_steps,
                observation_space=obs_space,
                action_space=act_space,
                lstm_hidden_size=LSTM_CONFIG["lstm_hidden_size"],
                lstm_num_layers=LSTM_CONFIG["lstm_num_layers"],
                device=device,
                n_envs=n_envs,
            )
            
            # Crea environments
            envs = [ScopaEnv(opponent_mode="random") for _ in range(n_envs)]
            
            # Initialize
            observations = np.stack([env.reset()[0] for env in envs])
            lstm_states = policy.get_initial_state(n_envs)
            episode_starts = np.ones(n_envs, dtype=np.float32)
            
            # Warm up
            for _ in range(2):
                buffer.reset()
                for step in range(n_steps):
                    action_masks = np.stack([env.action_masks() for env in envs])
                    with torch.no_grad():
                        obs_t = torch.from_numpy(observations).float().to(device)
                        masks_t = torch.from_numpy(action_masks).bool().to(device)
                        ep_t = torch.from_numpy(episode_starts).bool().to(device)
                        actions, values, log_probs, lstm_states = policy.forward(
                            obs_t, lstm_states, ep_t, masks_t
                        )
                    
                    actions_np = actions.cpu().numpy()
                    for j, env in enumerate(envs):
                        obs, _, done, _, _ = env.step(actions_np[j])
                        if done:
                            obs, _ = env.reset()
                        observations[j] = obs
                    episode_starts = np.zeros(n_envs, dtype=np.float32)
            
            # Benchmark
            start_time = time.time()
            steps_done = 0
            
            while steps_done < timesteps:
                buffer.reset()
                
                for step in range(n_steps):
                    action_masks = np.stack([env.action_masks() for env in envs])
                    
                    with torch.no_grad():
                        obs_t = torch.from_numpy(observations).float().to(device)
                        masks_t = torch.from_numpy(action_masks).bool().to(device)
                        ep_t = torch.from_numpy(episode_starts).bool().to(device)
                        
                        actions, values, log_probs, new_states = policy.forward(
                            obs_t, lstm_states, ep_t, masks_t
                        )
                    
                    actions_np = actions.cpu().numpy()
                    values_np = values.cpu().numpy()
                    log_probs_np = log_probs.cpu().numpy()
                    
                    new_observations = []
                    rewards = np.zeros(n_envs, dtype=np.float32)
                    new_episode_starts = np.zeros(n_envs, dtype=np.float32)
                    
                    for j, env in enumerate(envs):
                        obs, reward, done, _, _ = env.step(actions_np[j])
                        rewards[j] = reward
                        if done:
                            obs, _ = env.reset()
                            new_episode_starts[j] = 1.0
                        new_observations.append(obs)
                    
                    buffer.add(
                        observations, actions_np, rewards, episode_starts,
                        values_np, log_probs_np,
                        (lstm_states[0].cpu().numpy(), lstm_states[1].cpu().numpy()),
                        action_masks,
                    )
                    
                    observations = np.stack(new_observations)
                    lstm_states = new_states
                    episode_starts = new_episode_starts
                
                # Compute GAE
                with torch.no_grad():
                    obs_t = torch.from_numpy(observations).float().to(device)
                    ep_t = torch.from_numpy(episode_starts).bool().to(device)
                    last_values = policy.get_values(obs_t, lstm_states, ep_t)
                
                buffer.compute_returns_and_advantage(last_values.cpu().numpy(), episode_starts)
                
                # PPO update (1 epoch)
                for batch in buffer.get(batch_size=min(512, n_steps * n_envs)):
                    values, log_probs, entropy = policy.evaluate_actions(
                        batch.observations, batch.actions,
                        batch.lstm_states, batch.episode_starts, batch.action_masks,
                    )
                    
                    ratio = torch.exp(log_probs - batch.old_log_probs)
                    policy_loss = -(batch.advantages * ratio).mean()
                    value_loss = ((values - batch.returns) ** 2).mean()
                    
                    loss = policy_loss + 0.5 * value_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                steps_done += n_steps * n_envs
            
            elapsed = time.time() - start_time
            speed = steps_done / elapsed
            
            # Cleanup
            for env in envs:
                env.close()
            del policy, buffer, optimizer
            if device == "cuda":
                torch.cuda.empty_cache()
            
            results[name] = speed
            print(f"✅ {speed:,.0f} steps/s")
            
        except Exception as e:
            print(f"❌ Errore: {e}")
            results[name] = 0
    
    # Risultati
    valid = {k: v for k, v in results.items() if v > 0}
    if not valid:
        print("\n❌ Nessun test completato!")
        return results
    
    best = max(valid, key=valid.get)
    baseline = results.get(" 1 env CPU", list(valid.values())[0])
    
    print("\n" + "=" * 70)
    print("📊 RISULTATI")
    print("=" * 70)
    print(f"{'Config':<20} {'Steps/s':>12} {'Speedup':>10}")
    print("-" * 42)
    
    for rank, (cfg, speed) in enumerate(sorted(results.items(), key=lambda x: -x[1]), 1):
        if speed > 0:
            marker = " 👑" if cfg == best else ""
            speedup = speed / baseline if baseline > 0 else 1.0
            print(f"{cfg:<20} {speed:>12,.0f} {speedup:>9.2f}x{marker}")
    
    print("\n" + "=" * 70)
    print(f"🏆 CONFIGURAZIONE OTTIMALE: {best.strip()} ({valid[best]:,.0f} steps/s)")
    print("=" * 70)
    
    # Genera comando ideale
    best_cfg = next((cfg for name, cfg in configs if name == best), None)
    if best_cfg:
        device_flag = f"--device {best_cfg['device']}"
        n_envs = best_cfg["n_envs"]
        print(f"\n💡 COMANDO IDEALE per il tuo sistema:")
        print(f"   python scripts/train_recurrent.py --timesteps 10_000_000 --n-envs {n_envs} {device_flag} --fresh")
        print("")
    
    return results


def train(
    total_timesteps: int,
    fresh: bool = False,
    device: str = "cpu",
    name: str = DEFAULT_MODEL_NAME,
    # Override da CLI (None = usa config)
    n_envs: Optional[int] = None,
    opponent: Optional[str] = None,
    n_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    eval_freq: Optional[int] = None,
    save_freq: Optional[int] = None,
) -> None:
    """
    Training loop per MaskableRecurrentPolicy.
    """
    ensure_dirs()
    
    # Merge config con override
    cfg = {
        "n_envs": n_envs or CONFIG["n_envs"],
        "opponent_mode": opponent or CONFIG["opponent_mode"],
        "n_steps": n_steps or CONFIG["n_steps"],
        "batch_size": batch_size or CONFIG["batch_size"],
        "n_epochs": CONFIG["n_epochs"],
        "lr_initial": CONFIG["lr_initial"],
        "lr_final": CONFIG["lr_final"],
        "gamma": CONFIG["gamma"],
        "gae_lambda": CONFIG["gae_lambda"],
        "clip_range": CONFIG["clip_range"],
        "max_grad_norm": CONFIG["max_grad_norm"],
        "ent_coef_initial": CONFIG["ent_coef_initial"],
        "ent_coef_final": CONFIG["ent_coef_final"],
        "eval_freq": eval_freq or CONFIG["eval_freq"],
        "eval_games": CONFIG["eval_games"],
        "save_freq": save_freq or CONFIG["save_freq"],
        "log_freq": CONFIG["log_freq"],
    }
    
    # Directories
    save_dir = MODELS_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = LOGS_DIR / "train_recurrent"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Log file
    log_path = log_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timesteps", "time_elapsed", "fps",
            "policy_loss", "value_loss", "entropy",
            "lr", "ent_coef",
            "wr_random", "wr_heuristic"
        ])
    
    # Banner
    print(f"\n{'='*60}")
    print("🚀 TRAINING RECURRENT POLICY")
    print(f"{'='*60}")
    print(f"Run name: {name}")
    print(f"Session steps: {total_timesteps:,}")
    print(f"Fresh start: {fresh}")
    print(f"Device: {device}")
    print(f"Envs: {cfg['n_envs']} | Steps: {cfg['n_steps']} | Batch: {cfg['batch_size']}")
    print(f"Opponent: {cfg['opponent_mode']}")
    print(f"Save dir: {save_dir}")
    print(f"{'='*60}\n")
    
    # Create environments
    import gymnasium as gym
    obs_space = gym.spaces.Box(low=0, high=1, shape=(296,), dtype=np.float32)
    act_space = gym.spaces.Discrete(40)
    
    envs = [ScopaEnv(opponent_mode=cfg["opponent_mode"]) for _ in range(cfg["n_envs"])]
    
    # Create policy
    policy = MaskableRecurrentPolicy(
        obs_space, act_space,
        features_dim=LSTM_CONFIG["features_dim"],
        lstm_hidden_size=LSTM_CONFIG["lstm_hidden_size"],
        lstm_num_layers=LSTM_CONFIG["lstm_num_layers"],
    ).to(device)
    
    # Load checkpoint se non fresh
    start_timesteps = 0
    if not fresh:
        start_timesteps = load_checkpoint(policy, save_dir)
    
    # total_timesteps è la durata della SESSIONE, non il target totale
    # Calcola il target finale come: start + session
    target_timesteps = start_timesteps + total_timesteps
    
    if start_timesteps > 0:
        print(f"▶️ Continuo da {start_timesteps:,} steps")
        print(f"   Sessione: +{total_timesteps:,} → target: {target_timesteps:,} steps")
    else:
        print("▶️ Training da zero")
    
    optimizer = optim.Adam(policy.parameters(), lr=cfg["lr_initial"])
    
    # Create buffer
    buffer = RecurrentRolloutBuffer(
        buffer_size=cfg["n_steps"],
        observation_space=obs_space,
        action_space=act_space,
        lstm_hidden_size=LSTM_CONFIG["lstm_hidden_size"],
        lstm_num_layers=LSTM_CONFIG["lstm_num_layers"],
        device=device,
        gae_lambda=cfg["gae_lambda"],
        gamma=cfg["gamma"],
        n_envs=cfg["n_envs"],
    )
    
    # Initialize
    observations = np.stack([env.reset()[0] for env in envs])
    lstm_states = policy.get_initial_state(cfg["n_envs"])
    episode_starts = np.ones(cfg["n_envs"], dtype=np.float32)
    
    timesteps = start_timesteps
    start_time = time.time()
    best_wr_heuristic = 0.0
    
    # Training metrics
    recent_policy_loss = []
    recent_value_loss = []
    recent_entropy = []
    
    while timesteps < target_timesteps:
        progress = (timesteps - start_timesteps) / total_timesteps
        
        # Current hyperparameters
        current_lr = linear_schedule(cfg["lr_initial"], cfg["lr_final"], progress)
        current_ent_coef = linear_schedule(cfg["ent_coef_initial"], cfg["ent_coef_final"], progress)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Collect rollout
        buffer.reset()
        
        for step in range(cfg["n_steps"]):
            action_masks = np.stack([env.action_masks() for env in envs])
            
            with torch.no_grad():
                obs_t = torch.from_numpy(observations).float().to(device)
                masks_t = torch.from_numpy(action_masks).bool().to(device)
                episode_starts_t = torch.from_numpy(episode_starts).bool().to(device)
                
                actions, values, log_probs, new_lstm_states = policy.forward(
                    obs_t, lstm_states, episode_starts_t, masks_t
                )
            
            actions_np = actions.cpu().numpy()
            values_np = values.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()
            
            # Step environments
            new_observations = []
            rewards = np.zeros(cfg["n_envs"], dtype=np.float32)
            new_episode_starts = np.zeros(cfg["n_envs"], dtype=np.float32)
            
            for i, env in enumerate(envs):
                obs, reward, done, truncated, _ = env.step(actions_np[i])
                rewards[i] = reward
                
                if done or truncated:
                    obs, _ = env.reset()
                    new_episode_starts[i] = 1.0
                
                new_observations.append(obs)
            
            buffer.add(
                observations, actions_np, rewards, episode_starts,
                values_np, log_probs_np,
                (lstm_states[0].cpu().numpy(), lstm_states[1].cpu().numpy()),
                action_masks,
            )
            
            observations = np.stack(new_observations)
            lstm_states = new_lstm_states
            episode_starts = new_episode_starts
            
            # Reset LSTM states for done envs
            done_indices = np.where(new_episode_starts > 0.5)[0]
            if len(done_indices) > 0:
                new_h = lstm_states[0].clone()
                new_c = lstm_states[1].clone()
                new_h[:, done_indices, :] = 0.0
                new_c[:, done_indices, :] = 0.0
                lstm_states = (new_h, new_c)
        
        # Compute returns
        with torch.no_grad():
            obs_t = torch.from_numpy(observations).float().to(device)
            episode_starts_t = torch.from_numpy(episode_starts).bool().to(device)
            last_values = policy.get_values(obs_t, lstm_states, episode_starts_t)
            last_values_np = last_values.cpu().numpy()
        
        buffer.compute_returns_and_advantage(last_values_np, episode_starts)
        
        # PPO update
        for epoch in range(cfg["n_epochs"]):
            for batch in buffer.get(batch_size=cfg["batch_size"]):
                values, log_probs, entropy = policy.evaluate_actions(
                    batch.observations, batch.actions,
                    batch.lstm_states, batch.episode_starts, batch.action_masks,
                )
                
                log_ratio = log_probs - batch.old_log_probs
                ratio = torch.exp(log_ratio)
                
                policy_loss_1 = batch.advantages * ratio
                policy_loss_2 = batch.advantages * torch.clamp(
                    ratio, 1 - cfg["clip_range"], 1 + cfg["clip_range"]
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                value_loss = ((values - batch.returns) ** 2).mean()
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + 0.5 * value_loss + current_ent_coef * entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
                optimizer.step()
                
                recent_policy_loss.append(policy_loss.item())
                recent_value_loss.append(value_loss.item())
                recent_entropy.append(entropy.mean().item())
        
        timesteps += cfg["n_steps"] * cfg["n_envs"]
        
        # Logging
        if timesteps % cfg["log_freq"] < cfg["n_steps"] * cfg["n_envs"]:
            elapsed = time.time() - start_time
            fps = (timesteps - start_timesteps) / elapsed
            
            avg_p_loss = np.mean(recent_policy_loss[-100:]) if recent_policy_loss else 0
            avg_v_loss = np.mean(recent_value_loss[-100:]) if recent_value_loss else 0
            avg_entropy = np.mean(recent_entropy[-100:]) if recent_entropy else 0
            
            print(f"📊 {timesteps:,} steps | "
                  f"p_loss: {avg_p_loss:.4f} | v_loss: {avg_v_loss:.2f} | "
                  f"ent: {avg_entropy:.2f} | ent_coef: {current_ent_coef:.3f} | fps: {fps:.0f}")
        
        # Evaluation
        if timesteps % cfg["eval_freq"] < cfg["n_steps"] * cfg["n_envs"]:
            print(f"\n🎯 Evaluating at {timesteps:,} steps...")
            
            metrics_random = evaluate_policy(policy, "random", cfg["eval_games"])
            metrics_heuristic = evaluate_policy(policy, "heuristic", cfg["eval_games"])
            
            wr_random = metrics_random["winrate"]
            wr_heuristic = metrics_heuristic["winrate"]
            
            print(f"   vs random: {wr_random:.1%}")
            print(f"   vs heuristic: {wr_heuristic:.1%}")
            
            # Log to CSV
            elapsed = time.time() - start_time
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timesteps, elapsed, (timesteps - start_timesteps) / elapsed,
                    np.mean(recent_policy_loss[-100:]) if recent_policy_loss else 0,
                    np.mean(recent_value_loss[-100:]) if recent_value_loss else 0,
                    np.mean(recent_entropy[-100:]) if recent_entropy else 0,
                    current_lr, current_ent_coef,
                    wr_random, wr_heuristic
                ])
            
            if wr_heuristic > best_wr_heuristic:
                best_wr_heuristic = wr_heuristic
                torch.save(policy.state_dict(), save_dir / "best_vs_heuristic.pt")
                print(f"   ⭐ New best! Saved.")
            print()
        
        # Periodic save
        if timesteps % cfg["save_freq"] < cfg["n_steps"] * cfg["n_envs"]:
            checkpoint_name = f"checkpoint_{timesteps // 1_000_000}M.pt"
            torch.save(policy.state_dict(), save_dir / checkpoint_name)
            print(f"💾 Saved {checkpoint_name}")
    
    # Final save
    torch.save(policy.state_dict(), save_dir / "final.pt")
    
    for env in envs:
        env.close()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Training complete!")
    print(f"   Total time: {elapsed/3600:.1f} hours")
    print(f"   Best WR vs heuristic: {best_wr_heuristic:.1%}")
    print(f"   Models saved to: {save_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Train MaskableRecurrentPolicy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark modello corrente
  python scripts/train_recurrent.py --benchmark
  
  # Benchmark HARDWARE (trova config ottimale)
  python scripts/train_recurrent.py --hw-benchmark
  
  # Training 10M steps da zero
  python scripts/train_recurrent.py -t 10_000_000 --fresh
  
  # Continua training esistente fino a 20M
  python scripts/train_recurrent.py -t 20_000_000
  
  # Con GPU e custom opponent
  python scripts/train_recurrent.py -t 10_000_000 --device cuda --opponent random
"""
    )
    
    # Required
    parser.add_argument("--timesteps", "-t", type=int, default=10_000_000,
                        help="Numero totale di timesteps (default: 10M)")
    
    # Modes
    parser.add_argument("--benchmark", action="store_true",
                        help="Solo valuta modello corrente (no training)")
    parser.add_argument("--hw-benchmark", action="store_true",
                        help="Benchmark hardware per trovare config ottimale")
    parser.add_argument("--fresh", action="store_true",
                        help="Inizia da zero (ignora checkpoint esistenti)")
    
    # Device
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device per training (default: cpu)")
    
    # Run name
    parser.add_argument("--name", type=str, default=DEFAULT_MODEL_NAME,
                        help=f"Nome del run (default: {DEFAULT_MODEL_NAME})")
    
    # Override config
    parser.add_argument("--n-envs", type=int, default=None,
                        help=f"Numero ambienti paralleli (default: {CONFIG['n_envs']})")
    parser.add_argument("--opponent", type=str, default=None,
                        choices=["random", "heuristic", "self"],
                        help=f"Tipo avversario (default: {CONFIG['opponent_mode']})")
    parser.add_argument("--n-steps", type=int, default=None,
                        help=f"Steps per rollout (default: {CONFIG['n_steps']})")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Batch size (default: {CONFIG['batch_size']})")
    parser.add_argument("--eval-freq", type=int, default=None,
                        help=f"Frequenza valutazione (default: {CONFIG['eval_freq']})")
    parser.add_argument("--save-freq", type=int, default=None,
                        help=f"Frequenza checkpoint (default: {CONFIG['save_freq']})")
    
    args = parser.parse_args()
    
    # Benchmark mode (model evaluation)
    if args.benchmark:
        run_benchmark(
            name=args.name,
            device=args.device,
        )
        return
    
    # Hardware benchmark mode
    if args.hw_benchmark:
        run_hw_benchmark(timesteps=30_000)
        return
    
    # Training mode
    train(
        total_timesteps=args.timesteps,
        fresh=args.fresh,
        device=args.device,
        name=args.name,
        n_envs=args.n_envs,
        opponent=args.opponent,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
    )


if __name__ == "__main__":
    main()
