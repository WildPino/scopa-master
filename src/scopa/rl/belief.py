"""
BeliefNet - Rete per stimare le carte dell'avversario.

Stima la probabilità per ogni carta (40 totali) di essere nella mano
dell'avversario, basandosi sull'osservazione corrente e sulla storia.

Training: supervised learning su dati di self-play.
Metriche: BCE loss, Brier score, Top-k accuracy, calibration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Constants
NUM_CARDS = 40
OBSERVATION_DIM = 296


@dataclass
class BeliefMetrics:
    """Metriche di valutazione per BeliefNet."""
    bce_loss: float
    brier_score: float
    log_loss: float
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "bce_loss": round(self.bce_loss, 4),
            "brier_score": round(self.brier_score, 4),
            "log_loss": round(self.log_loss, 4),
            "top1_accuracy": round(self.top1_accuracy, 4),
            "top3_accuracy": round(self.top3_accuracy, 4),
            "top5_accuracy": round(self.top5_accuracy, 4),
        }
    
    def __str__(self) -> str:
        return (f"BCE: {self.bce_loss:.4f} | Brier: {self.brier_score:.4f} | "
                f"Top-1: {self.top1_accuracy:.2%} | Top-3: {self.top3_accuracy:.2%}")


class BeliefNet(nn.Module):
    """
    Rete per stimare la distribuzione di probabilità delle carte avversarie.
    
    Input: observation (296 dim) che include stato, history, e starter flag
    Output: per-card probability (40 dim) via sigmoid
    
    Architecture:
    - Input projection
    - 2-layer LSTM per processare la storia implicita
    - MLP finale con sigmoid output
    """
    
    def __init__(
        self,
        input_dim: int = OBSERVATION_DIM,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # LSTM for sequence processing (operates on history portion)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # concat input + lstm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_CARDS),
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: [batch, input_dim] observation tensor
            
        Returns:
            probs: [batch, 40] probability per card (sigmoid output)
        """
        batch_size = obs.shape[0]
        
        # Project input
        x = self.input_proj(obs)  # [batch, hidden]
        
        # LSTM processing (treat as single step for now)
        # In future: extract history and process as sequence
        x_seq = x.unsqueeze(1)  # [batch, 1, hidden]
        lstm_out, _ = self.lstm(x_seq)  # [batch, 1, hidden]
        lstm_feat = lstm_out.squeeze(1)  # [batch, hidden]
        
        # Concatenate original features and LSTM output
        combined = torch.cat([x, lstm_feat], dim=-1)  # [batch, hidden*2]
        
        # Output
        logits = self.output_mlp(combined)  # [batch, 40]
        probs = torch.sigmoid(logits)
        
        return probs
    
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Inference wrapper for numpy input.
        
        Args:
            obs: [obs_dim] or [batch, obs_dim] observation
            
        Returns:
            probs: [40] or [batch, 40] probabilities
        """
        self.eval()
        was_1d = obs.ndim == 1
        if was_1d:
            obs = obs[np.newaxis, :]
        
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float()
            if next(self.parameters()).is_cuda:
                obs_t = obs_t.cuda()
            probs = self.forward(obs_t).cpu().numpy()
        
        if was_1d:
            probs = probs[0]
        
        return probs


class BeliefDataset(Dataset):
    """
    Dataset per training supervisionato di BeliefNet.
    
    Ogni sample contiene:
    - observation: stato del gioco dal punto di vista dell'AI
    - target: binary vector [40] indicando quali carte ha l'avversario
    """
    
    def __init__(self, data: List[Dict[str, Any]]):
        """
        Args:
            data: lista di dict con 'observation' e 'opponent_hand_mask'
        """
        self.observations = []
        self.targets = []
        
        for item in data:
            self.observations.append(np.array(item['observation'], dtype=np.float32))
            self.targets.append(np.array(item['opponent_hand_mask'], dtype=np.float32))
        
        self.observations = np.stack(self.observations)
        self.targets = np.stack(self.targets)
    
    def __len__(self) -> int:
        return len(self.observations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.observations[idx]),
            torch.from_numpy(self.targets[idx]),
        )
    
    @classmethod
    def from_json(cls, path: str) -> "BeliefDataset":
        """Carica dataset da file JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(data)


class BeliefTrainer:
    """Pipeline per training supervisionato di BeliefNet."""
    
    def __init__(
        self,
        model: BeliefNet,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
    
    def compute_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> BeliefMetrics:
        """
        Calcola tutte le metriche di valutazione.
        
        Args:
            preds: [batch, 40] predicted probabilities
            targets: [batch, 40] binary targets
        """
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        # BCE Loss
        eps = 1e-7
        bce = -np.mean(
            targets_np * np.log(preds_np + eps) +
            (1 - targets_np) * np.log(1 - preds_np + eps)
        )
        
        # Brier Score: mean squared error between probs and binary targets
        brier = np.mean((preds_np - targets_np) ** 2)
        
        # Log Loss (per-sample average)
        log_loss = bce  # Same as BCE for binary
        
        # Top-k accuracy: check if top-k predicted cards contain a true card
        top1_acc = self._topk_accuracy(preds_np, targets_np, k=1)
        top3_acc = self._topk_accuracy(preds_np, targets_np, k=3)
        top5_acc = self._topk_accuracy(preds_np, targets_np, k=5)
        
        return BeliefMetrics(
            bce_loss=float(bce),
            brier_score=float(brier),
            log_loss=float(log_loss),
            top1_accuracy=float(top1_acc),
            top3_accuracy=float(top3_acc),
            top5_accuracy=float(top5_acc),
        )
    
    def _topk_accuracy(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        k: int
    ) -> float:
        """
        Calcola accuracy: quante volte almeno una delle top-k prediction
        è una carta effettivamente in mano all'avversario.
        """
        correct = 0
        for pred, target in zip(preds, targets):
            # Get indices of top-k predictions
            topk_idx = np.argsort(pred)[-k:]
            # Check if any of them is in target
            if np.any(target[topk_idx] > 0.5):
                correct += 1
        return correct / len(preds) if len(preds) > 0 else 0.0
    
    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> Tuple[float, BeliefMetrics]:
        """Train per una epoca."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for obs, target in dataloader:
            obs = obs.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(obs)
            
            # BCE loss
            loss = F.binary_cross_entropy(preds, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * obs.size(0)
            all_preds.append(preds)
            all_targets.append(target)
        
        avg_loss = total_loss / len(dataloader.dataset)
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.compute_metrics(all_preds, all_targets)
        
        return avg_loss, metrics
    
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Tuple[float, BeliefMetrics]:
        """Valutazione su validation set."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for obs, target in dataloader:
                obs = obs.to(self.device)
                target = target.to(self.device)
                
                preds = self.model(obs)
                loss = F.binary_cross_entropy(preds, target)
                
                total_loss += loss.item() * obs.size(0)
                all_preds.append(preds)
                all_targets.append(target)
        
        avg_loss = total_loss / len(dataloader.dataset)
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.compute_metrics(all_preds, all_targets)
        
        return avg_loss, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        early_stop_patience: int = 10,
        save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Training loop completo con early stopping.
        
        Returns:
            history: dict con training/validation losses e metriche
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_brier": [],
            "val_brier": [],
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            history["train_brier"].append(train_metrics.brier_score)
            
            if val_loader is not None:
                val_loss, val_metrics = self.evaluate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_brier"].append(val_metrics.brier_score)
                
                # LR scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | "
                          f"Train: {train_metrics} | "
                          f"Val Brier: {val_metrics.brier_score:.4f}")
                
                if patience_counter >= early_stop_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Train: {train_metrics}")
        
        return history


def generate_belief_data_from_game(
    env,
    num_games: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Genera dati di training per BeliefNet da partite.
    
    Per ogni stato durante la partita, salva:
    - observation: l'osservazione dell'AI
    - opponent_hand_mask: binary [40] con 1 per carte in mano all'avversario
    
    Args:
        env: ScopaEnv instance
        num_games: numero di partite da generare
        
    Returns:
        List di samples per BeliefDataset
    """
    from scopa.rl import ScopaEnv
    
    data = []
    
    for game_idx in range(num_games):
        obs, _ = env.reset()
        done = False
        
        while not done:
            # Crea il target: quali carte ha l'avversario
            opponent_hand = env.engine.hands[1]
            opponent_mask = np.zeros(NUM_CARDS, dtype=np.float32)
            for card in opponent_hand:
                opponent_mask[card.index] = 1.0
            
            # Salva sample
            data.append({
                'observation': obs.tolist(),
                'opponent_hand_mask': opponent_mask.tolist(),
            })
            
            # Fai mossa random
            mask = env.action_masks()
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            action = np.random.choice(valid)
            obs, _, done, truncated, _ = env.step(action)
            done = done or truncated
        
        if (game_idx + 1) % 100 == 0:
            print(f"Generated {game_idx + 1}/{num_games} games, {len(data)} samples")
    
    return data


if __name__ == "__main__":
    # Test forward pass
    print("Testing BeliefNet forward pass...")
    model = BeliefNet()
    x = torch.randn(4, OBSERVATION_DIM)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    assert out.shape == (4, NUM_CARDS), f"Expected (4, {NUM_CARDS}), got {out.shape}"
    print("✅ BeliefNet OK!")
