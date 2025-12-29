# Scopa AI 🃏

Agente di **Reinforcement Learning** per il gioco di carte italiano della **Scopa**, basato su architettura **LSTM ricorrente** con **action masking**.

## 📋 Caratteristiche

- **MaskableRecurrentPolicy**: Policy LSTM con action masking per mosse legali
- **Sistema a due fasi**: selezione carta + selezione presa iterativa
- **Training parallelo**: Supporto multi-ambiente per training accelerato
- **Reward Shaping**: Sistema di reward intermedi configurabile per migliorare l'apprendimento
- **MCTS**: Monte Carlo Tree Search per inference avanzata
- **GUI Pygame**: Interfaccia grafica per giocare contro l'AI
- **Visualizzazione**: Plot automatici dei progressi di training

## 🧠 Architettura

### Policy Ricorrente (LSTM)
L'agente utilizza una **MaskableRecurrentActorCriticPolicy** che combina:
- **LSTM a 2 layer** (256 hidden units) per memoria temporale
- **Features extractor** (512 dim) per processare le osservazioni
- **Action masking** per mascherare azioni illegali

### Observation Space (296 dimensioni)

| Indici | Descrizione |
|--------|-------------|
| 0-39 | Mano AI (one-hot) |
| 40-79 | Tavolo |
| 80-119 | Prese AI |
| 120-159 | Prese avversario |
| 160-199 | Carte giocate avversario |
| 200-239 | Carte selezionate (Fase 1) |
| 240 | Mazzo rimanente (normalizzato) |
| 241-251 | Statistiche partita |
| 252-254 | Metadati fase |
| 255-294 | History buffer (ultime 40 carte giocate) |
| 295 | Starter player |

### Reward Shaping
Sistema configurabile di reward intermedi:
- **Bonus presa**: +0.2 per ogni cattura
- **Settebello**: +1.5 per catturare il 7♦
- **Denari/Sette**: +0.3/+0.2 per carte strategiche
- **Scopa**: +3.0 per svuotare il tavolo
- **Fine partita**: Bonus vittoria + differenza punti

## 🚀 Installazione

```bash
# Clona e installa in modalità development
cd scopa-master
pip install -e .

# Con dipendenze dev (pytest, black, ruff)
pip install -e ".[dev]"
```

## 💻 Utilizzo

### Training

```bash
# Training con policy ricorrente (raccomandato)
python scripts/train_recurrent.py --timesteps 10000000

# Training da zero (ignora checkpoint)
python scripts/train_recurrent.py --fresh --timesteps 5000000

# Con GPU
python scripts/train_recurrent.py --gpu

# Benchmark hardware
python scripts/train_recurrent.py --hw-benchmark

# Valutazione modello esistente
python scripts/train_recurrent.py --benchmark
```

**Flags disponibili:**
| Flag | Descrizione |
|------|-------------|
| `--timesteps` | Numero totale di timesteps |
| `--fresh` | Ignora checkpoint, training da zero |
| `--gpu` | Usa CUDA se disponibile |
| `--n-envs` | Numero ambienti paralleli |
| `--opponent` | Tipo avversario: random, heuristic, self, mixed |
| `--benchmark` | Solo valutazione, no training |
| `--hw-benchmark` | Test configurazione hardware ottimale |

### Giocare vs AI

```bash
# GUI Pygame
python -m scopa.gui

# Visualizzazione CLI (debug)
python -m scopa.cli.visual_match

# MCTS Play (avanzato)
python -m scopa.cli.mcts_play
```

### Valutazione

```bash
# Match di valutazione vs diversi avversari
python scripts/eval_match.py
```

### Visualizzazione Training

```bash
# Genera grafici dei progressi
python -m scopa.training.visualization
```

## 📁 Struttura Progetto

```
scopa-master/
├── pyproject.toml              # Packaging e dipendenze
├── scripts/
│   ├── train_recurrent.py      # Training loop LSTM
│   └── eval_match.py           # Script valutazione
├── src/scopa/
│   ├── config.py               # Configurazione centralizzata
│   ├── game/
│   │   ├── cards.py            # Card, Suit
│   │   └── engine.py           # ScopaEngine
│   ├── rl/
│   │   ├── environment.py      # ScopaEnv (Gymnasium)
│   │   ├── policies.py         # MaskableRecurrentPolicy
│   │   ├── networks.py         # Feature extractors
│   │   ├── belief.py           # Belief state tracking
│   │   └── determinize.py      # Determinization utilities
│   ├── search/
│   │   └── mcts.py             # Monte Carlo Tree Search
│   ├── training/
│   │   ├── rollout_buffer.py   # RecurrentRolloutBuffer
│   │   ├── callbacks.py        # Training callbacks
│   │   └── visualization.py    # Plot progressi
│   ├── gui/                    # Interfaccia Pygame
│   └── cli/                    # Tool CLI
├── assets/
│   └── carte_napoletane/       # Immagini carte
├── models/                     # Modelli addestrati
├── logs/                       # Log training
└── graphs/                     # Grafici generati
```

## ⚙️ Configurazione

La configurazione centralizzata si trova in `src/scopa/config.py`:

```python
# LSTM Architecture
LSTM_CONFIG = {
    "features_dim": 512,
    "lstm_hidden_size": 256,
    "lstm_num_layers": 2,
}

# Training
RECURRENT_TRAINING_CONFIG = {
    "n_envs": 16,
    "n_steps": 1024,
    "batch_size": 256,
    "lr_initial": 3e-4,
    "lr_final": 5e-5,
    "ent_coef_initial": 0.1,
    "ent_coef_final": 0.01,
    # ...
}

# Reward Shaping (toggle con USE_REWARD_SHAPING)
REWARD_SHAPING_CONFIG = {
    "capture_base": 0.2,
    "settebello": 1.5,
    "denari": 0.3,
    "scopa": 3.0,
    # ...
}
```

## 🎮 Regole della Scopa

- Mazzo da **40 carte** (4 semi × 10 ranghi)
- Se c'è una carta di rank uguale sul tavolo, **devi** prenderla
- Altrimenti puoi prendere combinazioni che sommano al rank
- **Scopa**: 1 punto extra quando svuoti il tavolo
- **Punti finali**: Carte (>20), Denari (>5), Settebello (7♦), Primiera

## �️ Development

```bash
# Formattazione
black src/ scripts/

# Linting
ruff check src/ scripts/

# Test (se disponibili)
pytest tests/
```

## 📊 Dipendenze

- `torch` - Neural networks
- `stable-baselines3` - PPO base
- `sb3-contrib` - MaskablePPO utilities
- `gymnasium` - Environment interface
- `pygame` - GUI
- `matplotlib` - Visualizzazione
- `numpy` - Numerical computing

## 📜 License

MIT
