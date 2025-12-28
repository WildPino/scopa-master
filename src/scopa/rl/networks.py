"""
ScopaNet - Architettura Neurale Custom per Scopa AI.

NOTA: Questo file è un LABORATORIO per future implementazioni custom.
Attualmente il training usa MlpPolicy di SB3 con policy_kwargs in train_parallel.py.

Questo modello può essere usato come base per:
- Layer convoluzionali per analizzare sequenze di carte
- Attention mechanisms per focus su carte strategiche
- Architetture più complesse che richiedono forward() custom
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from scopa.config import OBSERVATION_DIM, ACTION_DIM


class ScopaNet(nn.Module):
    """
    Rete neurale Actor-Critic per Scopa.
    
    Architettura:
    - Shared backbone (256-256-128)
    - Actor head (policy logits)
    - Critic head (value estimate)
    
    Supporta action masking per azioni illegali.
    """
    
    def __init__(
        self,
        input_dim: int = OBSERVATION_DIM,
        action_dim: int = ACTION_DIM,
        hidden_sizes: Tuple[int, ...] = (256, 256, 128)
    ):
        """
        Inizializza la rete.
        
        Args:
            input_dim: Dimensione dell'osservazione (default 255)
            action_dim: Numero di azioni possibili (default 40)
            hidden_sizes: Dimensioni dei layer nascosti
        """
        super().__init__()
        
        # Tronco comune (Shared Backbone)
        layers = []
        prev_size = input_dim
        for size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU()
            ])
            prev_size = size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor (Policy Head)
        self.actor = nn.Linear(prev_size, action_dim)
        
        # Critic (Value Head)
        self.critic = nn.Linear(prev_size, 1)
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass della rete.
        
        Args:
            x: Osservazione [batch, input_dim]
            mask: Maschera booleana azioni legali [batch, action_dim]
            
        Returns:
            Tuple (action_probs, value)
        """
        # Shared backbone
        hidden = self.shared_layers(x)
        
        # Policy logits
        policy_logits = self.actor(hidden)
        
        # Action masking
        if mask is not None:
            # Imposta -inf per azioni illegali
            fill_value = torch.finfo(policy_logits.dtype).min
            policy_logits = policy_logits.masked_fill(~mask, fill_value)
        
        # Probabilità (softmax)
        probs = F.softmax(policy_logits, dim=-1)
        
        # Value estimate
        value = self.critic(hidden)
        
        return probs, value
    
    def get_action(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Campiona un'azione dalla policy.
        
        Args:
            x: Osservazione
            mask: Maschera azioni legali
            deterministic: Se True, sceglie l'azione con prob massima
            
        Returns:
            Tuple (action, log_prob, value)
        """
        probs, value = self.forward(x, mask)
        
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        log_prob = torch.log(probs.gather(1, action.unsqueeze(-1))).squeeze(-1)
        
        return action, log_prob, value.squeeze(-1)
