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

# === TRAINING HYPERPARAMETERS ===
DEFAULT_HYPERPARAMS: Dict[str, Any] = {
    "learning_rate": 0.0003,
    "learning_rate_final": 0.00005,
    "gamma": 0.99,
    "n_steps": 2048,
    "batch_size": 1024,
    "n_epochs": 10,
    "ent_coef": 0.07,
    "clip_range": 0.2,
}

# Architettura neural network (MlpPolicy)
NETWORK_ARCH = {
    "pi": [256, 256, 128],  # Actor
    "vf": [256, 256, 128],  # Critic
}

# === OPPONENT MODES ===
OPPONENT_MODES = ["random", "self", "heuristic", "mixed"]


def ensure_dirs() -> None:
    """Crea le directory di output se non esistono."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
