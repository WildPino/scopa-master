"""
Configurazione centralizzata per Scopa AI.

Contiene:
- Paths di default per modelli, log, assets
- Hyperparameters di training
- Costanti del gioco
"""
from pathlib import Path
from typing import Dict, Any

# === PATHS ===
# Root del progetto (dove si trova pyproject.toml)
# config.py → scopa/ → src/ → scopa-master/
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Directory assets
ASSETS_DIR = PROJECT_ROOT / "assets"
CARDS_DIR = ASSETS_DIR / "carte_napoletane"

# Directory output
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
GRAPHS_DIR = PROJECT_ROOT / "graphs"

# === GAME CONSTANTS ===
NUM_CARDS = 40
NUM_SUITS = 4
CARDS_PER_SUIT = 10
HAND_SIZE = 3
INITIAL_TABLE_SIZE = 4

# Valori Primiera (per calcolo punteggio)
PRIMIERA_VALUES: Dict[int, int] = {
    7: 21, 6: 18, 1: 16, 5: 15, 4: 14, 3: 13, 2: 12, 8: 10, 9: 10, 10: 10
}

# === RL ENVIRONMENT ===
OBSERVATION_DIM = 296  # 255 base + 40 history buffer + 1 starter player
HISTORY_BUFFER_SIZE = 40
ACTION_DIM = 40

#DEPRECATED
# === TRAINING HYPERPARAMETERS ===
DEFAULT_HYPERPARAMS: Dict[str, Any] = {
    "learning_rate": 0.0003,
    "learning_rate_final": 0.00005,
    "gamma": 0.99,
    "n_steps": 2048,
    "batch_size": 1024,
    "n_epochs": 10,
    "ent_coef": 0.01, # Started with 0.07
    "clip_range": 0.2,
}

# Architettura neural network (MlpPolicy)
NETWORK_ARCH = {
    "pi": [256, 256, 128],  # Actor
    "vf": [256, 256, 128],  # Critic
}

# === OPPONENT MODES ===
OPPONENT_MODES = ["random", "self", "heuristic", "mixed"]

# === RECURRENT TRAINING CONFIG ===
# Tutti i parametri per train_recurrent.py
RECURRENT_TRAINING_CONFIG: Dict[str, Any] = {
    # Environment
    "n_envs": 16,
    "opponent_mode": "random", # TODO: "heuristic"
    
    # Rollout
    "n_steps": 2048,  # era 1024
    "batch_size": 512,
    "n_epochs": 10,
    
    # Learning rate (con decay lineare)
    "lr_initial": 3e-4, 
    "lr_final": 3e-5,
    
    # PPO
    "gamma": 0.95,  #0.99
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "max_grad_norm": 0.5,
    
    # Entropy (con decay lineare - ALTO per prevenire collapse)
    "ent_coef_initial": 0.5, #0.3  
    "ent_coef_final": 0.15, #0.05  
    
    # Evaluation
    "eval_freq": 100_000,
    "eval_games": 100,
    
    # Checkpoints
    "save_freq": 1_000_000,
    "log_freq": 10_000,
}

# === LSTM ARCHITECTURE ===
LSTM_CONFIG: Dict[str, Any] = {
    "features_dim": 512,
    "lstm_hidden_size": 256,
    "lstm_num_layers": 2,
}

# === REWARD SHAPING CONFIG ===
# Toggle: True = usa reward intermedi (training iniziale), False = solo reward finale (fine-tuning)
USE_REWARD_SHAPING: bool = False

REWARD_SHAPING_CONFIG: Dict[str, float] = {
    # Reward intermedi per azione
    "capture_base": 0.2,             # Bonus base per ogni presa
    "settebello": 1.5,               # Bonus prendere 7 di denari
    "denari": 0.3,                   # Bonus per ogni denaro preso
    "sette": 0.2,                    # Bonus per ogni 7 preso (primiera)
    "scopa": 3.0,                    # Bonus massiccio per scopa
    "calata_penalty": -0.05,         # Penalità leggera per calata (quando potevi prendere)
    
    # Reward finali (applicati sempre)
    "score_multiplier": 2.0,         # Moltiplica la differenza punti finale
    "win_bonus": 5.0,                # Bonus secco per vittoria
}

# === MCTS PROFILES ===
MCTS_PROFILES: Dict[str, Dict[str, int]] = {
    "tournament": {"num_det": 100, "sims_per_det": 100},  # < 5s
    "low_latency": {"num_det": 10, "sims_per_det": 20},   # < 500ms
    "training": {"num_det": 50, "sims_per_det": 30},
}

# === ENDGAME ===
ENDGAME_THRESHOLD = 6  # Attiva endgame solver se remaining_cards_per_player <= 6

# === DEFAULT PATHS ===
DEFAULT_MODEL_NAME = "lstm_baseline"


def ensure_dirs() -> None:
    """Crea le directory di output se non esistono."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
