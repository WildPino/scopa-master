"""
Train - Script di addestramento base per Scopa AI.

Usage:
    python -m scopa.training.train --mode random --timesteps 1000000
    python -m scopa.training.train --gpu --fresh
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Optional

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from scopa.rl import ScopaEnv
from scopa.config import (
    MODELS_DIR, LOGS_DIR, 
    DEFAULT_HYPERPARAMS, OPPONENT_MODES,
    ensure_dirs
)
from scopa.training.callbacks import SelfPlayCallback


def mask_fn(env: ScopaEnv):
    """Estrae action masks dall'ambiente."""
    return env.action_masks()


def train(
    total_timesteps: int = 1_000_000,
    opponent_mode: str = "random",
    continue_from: Optional[str] = None,
    use_gpu: bool = False
) -> None:
    """
    Addestra il modello MaskablePPO per la Scopa.
    
    Args:
        total_timesteps: Numero totale di step
        opponent_mode: Modalità avversario
        continue_from: Path modello esistente
        use_gpu: Se True usa CUDA
    """
    device = "cuda" if use_gpu else "cpu"
    print(f"🖥️  Device: {device.upper()}")
    
    ensure_dirs()
    
    # Crea ambiente
    env = ScopaEnv(opponent_mode=opponent_mode)
    env = ActionMasker(env, mask_fn)
    env = Monitor(env, str(LOGS_DIR))
    
    # Carica o crea modello
    hp = DEFAULT_HYPERPARAMS
    
    if continue_from:
        print(f"📂 Caricamento modello da: {continue_from}")
        model = MaskablePPO.load(
            continue_from, 
            env=env, 
            ent_coef=hp["ent_coef"],
            learning_rate=hp["learning_rate"],
            device=device
        )
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=hp["learning_rate"],
            gamma=hp["gamma"],
            n_steps=hp["n_steps"],
            batch_size=hp["batch_size"],
            n_epochs=hp["n_epochs"],
            ent_coef=hp["ent_coef"],
            device=device,
        )
    
    # Setup callbacks
    callbacks = []
    if opponent_mode in ["self", "mixed"]:
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        base_env.set_model(model)
        callbacks.append(SelfPlayCallback(update_freq=200_000, verbose=1))
    
    # Training
    mode_emoji = {"random": "🎲", "self": "🪞", "heuristic": "🧠", "mixed": "🔀"}
    print(f"\n{mode_emoji.get(opponent_mode, '❓')} Modalità: {opponent_mode.upper()}")
    print(f"🚀 Inizio addestramento per {total_timesteps:,} timestep...")
    
    model.learn(total_timesteps=total_timesteps, callback=callbacks or None)
    
    # Salva modello
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"scopa_ai_{opponent_mode}_{timestamp}"
    model.save(str(model_path))
    model.save(str(MODELS_DIR / f"scopa_ai_{opponent_mode}_latest"))
    model.save(str(MODELS_DIR / "scopa_ai_latest"))
    
    print(f"\n✅ Addestramento completato!")
    print(f"💾 Modello salvato: {model_path}.zip")


def main():
    """Entry point per CLI."""
    parser = argparse.ArgumentParser(description="Addestra Scopa AI")
    parser.add_argument(
        "--mode", "-m", type=str, default="random",
        choices=OPPONENT_MODES,
        help="Modalità avversario"
    )
    parser.add_argument(
        "--timesteps", "-t", type=int, default=1_000_000,
        help="Numero totale di timesteps"
    )
    parser.add_argument(
        "--continue-from", "-c", type=str,
        default=str(MODELS_DIR / "scopa_ai_latest"),
        help="Path modello da cui continuare"
    )
    parser.add_argument(
        "--fresh", "-f", action="store_true",
        help="Ignora modello esistente"
    )
    parser.add_argument(
        "--gpu", "-g", action="store_true",
        help="Usa GPU (CUDA)"
    )
    
    args = parser.parse_args()
    
    continue_from = args.continue_from
    if args.fresh:
        continue_from = None
        print("🆕 Avvio fresh (ignoro modelli esistenti)")
    elif continue_from and not os.path.exists(continue_from + ".zip"):
        print(f"⚠️  Modello non trovato: {continue_from}.zip")
        print("   Avvio nuovo addestramento da zero...")
        continue_from = None
    
    train(
        total_timesteps=args.timesteps,
        opponent_mode=args.mode,
        continue_from=continue_from,
        use_gpu=args.gpu,
    )


if __name__ == "__main__":
    main()
