"""
Networks - Architetture neurali legacy per Scopa AI.

NOTA: Questo modulo contiene componenti legacy mantenuti per compatibilità.
Il modello attivo usa MaskableRecurrentPolicy da policies.py.

Per nuovi sviluppi, usare:
- scopa.rl.policies.MaskableRecurrentPolicy

Componenti legacy (non più utilizzati attivamente):
- ScopaNet: Rete MLP base Actor-Critic (per vecchi modelli MaskablePPO)
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
    [LEGACY] Rete neurale Actor-Critic MLP per Scopa.
    
    Mantenuta per compatibilità con vecchi modelli MaskablePPO.
    Per nuovi training, usare MaskableRecurrentPolicy.
    
    Architettura:
    - Shared backbone (256-256-128)
    - Actor head (policy logits)
    - Critic head (value estimate)
    """
    
    def __init__(
        self,
        input_dim: int = OBSERVATION_DIM,
        action_dim: int = ACTION_DIM,
        hidden_sizes: Tuple[int, ...] = (256, 256, 128)
    ):
        super().__init__()
        
        # Shared backbone
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
        Forward pass.
        
        Args:
            x: Osservazione [batch, input_dim]
            mask: Maschera booleana azioni legali [batch, action_dim]
            
        Returns:
            Tuple (action_probs, value)
        """
        hidden = self.shared_layers(x)
        policy_logits = self.actor(hidden)
        
        if mask is not None:
            fill_value = torch.finfo(policy_logits.dtype).min
            policy_logits = policy_logits.masked_fill(~mask, fill_value)
        
        probs = F.softmax(policy_logits, dim=-1)
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
