#DEPRECATED
#Removing 40 confirmation button and implementing correct masking this is naturaly handled by the environment

"""
Train Phase 1 - Script di addestramento supervisionato per l'aritmetica di Fase 1

Questo script allena l'AI a capire quando la somma delle carte selezionate
è corretta rispetto al rank della carta giocata.

Modalità:
    1. collect: Raccoglie dati Fase 1 durante una sessione normale
    2. train: Allena un modello solo sui dati Fase 1 raccolti
"""
import os
import argparse
import numpy as np
from datetime import datetime
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from scopa_env import ScopaEnv, PHASE1_DATA_DIR

# Directory
MODEL_DIR = "./models/"


class Phase1DataCollectorCallback(BaseCallback):
    """
    Callback per salvare periodicamente i dati Fase 1.
    """
    def __init__(self, save_freq=10000, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.total_saved = 0
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Accedi all'ambiente base
            env = self.training_env.envs[0]
            if hasattr(env, 'env'):
                env = env.env
            
            if hasattr(env, 'save_phase1_buffer'):
                stats = env.get_phase1_stats()
                if stats['count'] > 0:
                    env.save_phase1_buffer()
                    self.total_saved += stats['count']
                    if self.verbose > 0:
                        print(f"[Phase1] Salvati {stats['count']} campioni "
                              f"({stats['correct']} corretti, {stats['wrong']} errati). "
                              f"Totale: {self.total_saved}")
        return True
    
    def _on_training_end(self):
        # Salva i dati rimanenti alla fine
        env = self.training_env.envs[0]
        if hasattr(env, 'env'):
            env = env.env
        
        if hasattr(env, 'save_phase1_buffer'):
            stats = env.get_phase1_stats()
            if stats['count'] > 0:
                env.save_phase1_buffer()
                self.total_saved += stats['count']
                print(f"[Phase1] Salvati ultimi {stats['count']} campioni. Totale finale: {self.total_saved}")


def collect_phase1_data(timesteps=100000, opponent_mode='random', continue_from=None):
    """
    Esegue una sessione di training raccogliendo dati Fase 1.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Crea ambiente con raccolta dati Fase 1 attiva
    env = ScopaEnv(opponent_mode=opponent_mode, save_phase1_data=True)
    env = Monitor(env, "./logs/")
    
    # Carica o crea modello
    if continue_from and os.path.exists(continue_from + '.zip'):
        print(f"📂 Caricamento modello da: {continue_from}")
        model = MaskablePPO.load(continue_from, env=env)
    else:
        print("🆕 Creazione nuovo modello...")
        model = MaskablePPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=0.0003,
            n_steps=2052,
            batch_size=256,
        )
    
    # Callback per salvare dati Fase 1
    callback = Phase1DataCollectorCallback(save_freq=10000, verbose=1)
    
    print(f"\n📊 Raccolta dati Fase 1 per {timesteps} timesteps...")
    print(f"   💾 Dati salvati in: {PHASE1_DATA_DIR}")
    
    model.learn(total_timesteps=timesteps, callback=callback)
    
    print(f"\n✅ Raccolta completata! Totale campioni: {callback.total_saved}")


def train_phase1_only(epochs=50, batch_size=64, learning_rate=0.001):
    """
    Allena un classificatore sull'aritmetica di Fase 1.
    Usa i dati raccolti per insegnare all'AI quando confermare.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    filepath = os.path.join(PHASE1_DATA_DIR, "phase1_data.npz")
    
    if not os.path.exists(filepath):
        print(f"❌ File dati non trovato: {filepath}")
        print("   Esegui prima: python train_phase1.py collect")
        return
    
    # Carica dati
    data = np.load(filepath)
    observations = data['observations']
    target_ranks = data['target_ranks']
    current_sums = data['current_sums']
    is_correct = data['is_correct']
    
    print(f"\n📊 Dati caricati:")
    print(f"   Totale campioni: {len(observations)}")
    print(f"   Corretti: {is_correct.sum()} ({100*is_correct.mean():.1f}%)")
    print(f"   Errati: {(~is_correct.astype(bool)).sum()} ({100*(1-is_correct.mean()):.1f}%)")
    
    # Prepara tensori
    X = torch.FloatTensor(observations)
    y = torch.FloatTensor(is_correct).unsqueeze(1)
    
    # Split train/val
    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    
    # Modello semplice per classificazione binaria
    class Phase1Classifier(nn.Module):
        def __init__(self, input_dim=255):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.net(x)
    
    model = Phase1Classifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\n🚀 Training per {epochs} epochs...")
    
    best_val_acc = 0
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val).item()
            val_acc = ((val_preds > 0.5) == y_val).float().mean().item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Salva miglior modello
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "phase1_classifier_best.pt"))
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}: Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.1f}%")
    
    print(f"\n✅ Training completato!")
    print(f"   Best Validation Accuracy: {best_val_acc*100:.1f}%")
    print(f"   Modello salvato in: {MODEL_DIR}phase1_classifier_best.pt")


def analyze_phase1_data():
    """
    Analizza i dati Fase 1 raccolti.
    """
    filepath = os.path.join(PHASE1_DATA_DIR, "phase1_data.npz")
    
    if not os.path.exists(filepath):
        print(f"❌ File non trovato: {filepath}")
        return
    
    data = np.load(filepath)
    observations = data['observations']
    target_ranks = data['target_ranks']
    current_sums = data['current_sums']
    is_correct = data['is_correct']
    
    print(f"\n📊 Analisi Dati Fase 1")
    print(f"=" * 40)
    print(f"Totale campioni: {len(observations)}")
    print(f"Corretti: {is_correct.sum()} ({100*is_correct.mean():.1f}%)")
    print(f"Errati: {(~is_correct.astype(bool)).sum()} ({100*(1-is_correct.mean()):.1f}%)")
    
    # Analisi per differenza
    diff = target_ranks - current_sums
    print(f"\n📉 Distribuzione errori (target - somma):")
    for d in range(-5, 6):
        count = ((diff == d) & ~is_correct.astype(bool)).sum()
        if count > 0:
            print(f"   Diff {d:+d}: {count} errori")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Fase 1 per Scopa AI')
    parser.add_argument('mode', choices=['collect', 'train', 'analyze'],
                        help='Modalità: collect (raccoglie dati), train (allena), analyze (analizza)')
    parser.add_argument('--timesteps', '-t', type=int, default=100000,
                        help='Timesteps per raccolta (default: 100000)')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Epochs per training (default: 50)')
    parser.add_argument('--continue-from', '-c', type=str, default='./models/scopa_ai_latest',
                        help='Modello da cui continuare per collect')
    
    args = parser.parse_args()
    
    if args.mode == 'collect':
        collect_phase1_data(
            timesteps=args.timesteps,
            continue_from=args.continue_from
        )
    elif args.mode == 'train':
        train_phase1_only(epochs=args.epochs)
    elif args.mode == 'analyze':
        analyze_phase1_data()
