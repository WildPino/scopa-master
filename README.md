# Scopa AI 🃏

Agente di **Reinforcement Learning** per il gioco di carte italiano della **Scopa**.

## 📋 Caratteristiche

- **MaskablePPO** con action masking per mosse legali
- **Sistema a due fasi**: selezione carta + selezione presa iterativa
- **Training parallelo** con SubprocVecEnv (8-16 ambienti)
- **Self-play** e modalità mixed per training avanzato
- **GUI Pygame** per giocare contro l'AI
- **Architettura 256-256-128** per policy e value network

## 🚀 Installazione

```bash
# Clona e installa in modalità development
cd scopa-master
pip install -e .
```

Oppure con dipendenze dev:
```bash
pip install -e ".[dev]"
```

## 💻 Utilizzo

### Training

```bash
# Training base (1M steps)
python -m scopa.training.train --mode random --timesteps 1000000

# Training parallelo ottimizzato (raccomandato)
python -m scopa.training.train_parallel --gpu --n-envs 8

# Benchmark per trovare configurazione ottimale
python -m scopa.training.train_parallel --benchmark
```

### Giocare vs AI

```bash
# GUI Pygame
python -m scopa.gui

# Visualizzazione CLI (debug)
python -m scopa.cli.visual_match
```

### Visualizzazione Training

```bash
python -m scopa.training.visualization
```

## 📁 Struttura Progetto

```
scopa/
├── pyproject.toml           # Packaging
├── src/
│   └── scopa/
│       ├── config.py        # Configurazione centralizzata
│       ├── game/            # Logica di gioco
│       │   ├── cards.py     # Card, Suit
│       │   └── engine.py    # ScopaEngine
│       ├── rl/              # Reinforcement Learning
│       │   ├── environment.py   # ScopaEnv (Gymnasium)
│       │   └── networks.py      # ScopaNet
│       ├── training/        # Script di training
│       │   ├── train.py
│       │   ├── train_parallel.py
│       │   └── visualization.py
│       ├── gui/             # Interfaccia Pygame
│       └── cli/             # Tool CLI
├── assets/
│   └── carte_napoletane/    # Immagini carte
├── models/                  # Modelli addestrati
└── logs/                    # Log training
```

## 🎮 Regole della Scopa

- Mazzo da 40 carte (4 semi × 10 ranghi)
- Se c'è una carta di rank uguale sul tavolo, **devi** prenderla
- Altrimenti puoi prendere combinazioni che sommano al rank
- **Scopa**: 1 punto extra quando svuoti il tavolo
- Punti finali: Carte (>20), Denari (>5), Settebello (7♦), Primiera

## 📊 Observation Space (255 dim)

| Indices | Descrizione |
|---------|-------------|
| 0-39 | Mano AI |
| 40-79 | Tavolo |
| 80-119 | Prese AI |
| 120-159 | Prese avversario |
| 160-199 | Carte giocate avversario |
| 200-239 | Carte selezionate (Fase 1) |
| 240 | Mazzo rimanente |
| 241-251 | Statistiche |
| 252-254 | Metadati fase |

## 🛠️ Development

```bash
# Formattazione
black src/

# Linting
ruff check src/

# Test
pytest tests/
```

## 📜 License

MIT
