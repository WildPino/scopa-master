"""
MaskableRecurrentPolicy - Policy ricorrente con action masking.

Combina:
- LSTM hidden states per memoria temporale
- Action masks per azioni illegali
- Compatibilità con SB3 training loop

Questo è il componente CRITICO che permette all'AI di apprendere.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import CategoricalDistribution

from scopa.config import OBSERVATION_DIM, ACTION_DIM, LSTM_CONFIG


class RecurrentFeaturesExtractor(nn.Module):
    """
    Features extractor con LSTM che propaga hidden states.
    
    A differenza di ScopaLSTMFeaturesExtractor (per inference),
    questo è progettato per training con gestione esplicita degli states.
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
    ):
        super().__init__()
        
        self.features_dim = features_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        input_dim = int(np.prod(observation_space.shape))
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, lstm_hidden_size),
            nn.ReLU(),
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        observations: Tensor,
        lstm_states: Tuple[Tensor, Tensor],
        episode_starts: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass con gestione hidden states.
        
        Args:
            observations: [batch, obs_dim] o [batch, seq, obs_dim]
            lstm_states: (h_n, c_n) where each is [n_layers, batch, hidden]
            episode_starts: [batch] boolean, True se episodio appena iniziato
            
        Returns:
            (features, new_lstm_states)
        """
        # Handle both single-step and sequence input
        if observations.dim() == 2:
            # Single step: [batch, obs_dim] -> [batch, 1, obs_dim]
            observations = observations.unsqueeze(1)
            single_step = True
        else:
            single_step = False
        
        batch_size, seq_len, _ = observations.shape
        
        # Reset hidden states where episode started
        h_n, c_n = lstm_states
        
        # episode_starts: [batch] -> need to broadcast to [n_layers, batch, hidden]
        if episode_starts.any():
            # Create mask: [n_layers, batch, hidden]
            reset_mask = episode_starts.view(1, batch_size, 1).expand_as(h_n)
            h_n = h_n.masked_fill(reset_mask, 0.0)
            c_n = c_n.masked_fill(reset_mask, 0.0)
        
        # Project input
        x = self.input_proj(observations)  # [batch, seq, lstm_hidden]
        
        # LSTM forward
        lstm_out, (new_h_n, new_c_n) = self.lstm(x, (h_n, c_n))
        
        # Project output
        features = self.output_proj(lstm_out)  # [batch, seq, features_dim]
        
        if single_step:
            features = features.squeeze(1)  # [batch, features_dim]
        
        return features, (new_h_n, new_c_n)
    
    def get_initial_state(self, batch_size: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Genera hidden states iniziali (zero)."""
        h_0 = torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_hidden_size,
            device=device
        )
        c_0 = torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_hidden_size,
            device=device
        )
        return (h_0, c_0)


class MaskableRecurrentActorCriticPolicy(nn.Module):
    """
    Policy Actor-Critic ricorrente con supporto action masks.
    
    Questa è l'implementazione core che:
    1. Mantiene LSTM hidden states per ogni ambiente
    2. Applica action masks agli output della policy
    3. È compatibile con PPO training
    
    API:
    - forward(obs, lstm_states, episode_starts, action_masks) 
        -> (actions, values, log_probs, new_states)
    - evaluate_actions(obs, actions, lstm_states, episode_starts, action_masks)
        -> (values, log_probs, entropy)
    - get_values(obs, lstm_states, episode_starts) -> values
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        features_dim: int = 256,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        ortho_init: bool = True,
    ):
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_dim = features_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # Features extractor con LSTM
        self.features_extractor = RecurrentFeaturesExtractor(
            observation_space=observation_space,
            features_dim=features_dim,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
        )
        
        # Actor (policy) head
        self.action_net = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_space.n),
        )
        
        # Critic (value) head
        self.value_net = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        # Inizializzazione ortogonale (come PPO)
        if ortho_init:
            self._init_weights()
    
    def _init_weights(self):
        """Inizializzazione ortogonale per stabilità."""
        for module in [self.action_net, self.value_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)
    
    def get_initial_state(self, batch_size: int = 1) -> Tuple[Tensor, Tensor]:
        """Ritorna hidden states iniziali."""
        device = next(self.parameters()).device
        return self.features_extractor.get_initial_state(batch_size, device)
    
    def forward(
        self,
        obs: Tensor,
        lstm_states: Tuple[Tensor, Tensor],
        episode_starts: Tensor,
        action_masks: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass completo (per rollout collection).
        
        Args:
            obs: [batch, obs_dim] observations
            lstm_states: (h_n, c_n) hidden states
            episode_starts: [batch] boolean
            action_masks: [batch, n_actions] boolean (True = valid)
            deterministic: se True, sceglie azione greedy
            
        Returns:
            (actions, values, log_probs, new_lstm_states)
        """
        # Extract features con LSTM
        features, new_lstm_states = self.features_extractor(
            obs, lstm_states, episode_starts
        )
        
        # Policy logits
        action_logits = self.action_net(features)
        
        # Apply action mask
        if action_masks is not None:
            # Mask invalid actions con -inf
            action_logits = action_logits.masked_fill(
                ~action_masks, float('-inf')
            )
        
        # Distribution
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        # Sample o greedy
        if deterministic:
            actions = probs.argmax(dim=-1)
        else:
            actions = dist.sample()
        
        # Log probs
        log_probs = dist.log_prob(actions)
        
        # Values
        values = self.value_net(features).squeeze(-1)
        
        return actions, values, log_probs, new_lstm_states
    
    def evaluate_actions(
        self,
        obs: Tensor,
        actions: Tensor,
        lstm_states: Tuple[Tensor, Tensor],
        episode_starts: Tensor,
        action_masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Valuta azioni date (per PPO update).
        
        Usa un path semplificato MLP-only durante il training:
        - L'observation contiene già history buffer (40 elementi)
        - L'LSTM è usato solo durante inference per hidden state propagation
        - Questo evita il problema dei gradienti zero per weight_hh
        
        Args:
            obs: [batch*seq, obs_dim] flattened observations
            actions: [batch*seq] flattened actions
            lstm_states: (ignorato durante training - usato solo per API compatibility)
            episode_starts: (ignorato durante training)
            action_masks: [batch*seq, n_actions] action masks
            
        Returns:
            (values, log_probs, entropy)
        """
        # Path semplificato: MLP-only con proiezione
        # L'observation history (features 255-294) fornisce contesto temporale
        features = self.features_extractor.input_proj(obs)
        features = self.features_extractor.output_proj(features)
        
        # Policy logits
        action_logits = self.action_net(features)
        
        # Apply mask
        if action_masks is not None:
            action_logits = action_logits.masked_fill(
                ~action_masks, float('-inf')
            )
        
        # Distribution
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        # Compute metrics
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Values
        values = self.value_net(features).squeeze(-1)
        
        return values, log_probs, entropy
    
    def get_values(
        self,
        obs: Tensor,
        lstm_states: Tuple[Tensor, Tensor],
        episode_starts: Tensor,
    ) -> Tensor:
        """Ottiene solo i values (per bootstrap)."""
        features, _ = self.features_extractor(obs, lstm_states, episode_starts)
        return self.value_net(features).squeeze(-1)
    
    def predict(
        self,
        obs: np.ndarray,
        lstm_states: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        episode_starts: Optional[np.ndarray] = None,
        action_masks: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Prediction wrapper per inference (numpy in/out).
        
        Args:
            obs: [obs_dim] o [batch, obs_dim]
            lstm_states: hidden states o None (inizializza)
            episode_starts: [batch] o None (assume False)
            action_masks: [n_actions] o [batch, n_actions]
            deterministic: se True, greedy
            
        Returns:
            (actions, new_lstm_states) come numpy
        """
        self.eval()
        
        # Handle dimensions
        was_1d = obs.ndim == 1
        if was_1d:
            obs = obs[np.newaxis, :]
            if action_masks is not None:
                action_masks = action_masks[np.newaxis, :]
        
        batch_size = obs.shape[0]
        device = next(self.parameters()).device
        
        # Convert to tensors
        obs_t = torch.from_numpy(obs).float().to(device)
        
        if lstm_states is None:
            lstm_states_t = self.get_initial_state(batch_size)
        else:
            lstm_states_t = (
                torch.from_numpy(lstm_states[0]).float().to(device),
                torch.from_numpy(lstm_states[1]).float().to(device),
            )
        
        if episode_starts is None:
            episode_starts_t = torch.zeros(batch_size, dtype=torch.bool, device=device)
        else:
            episode_starts_t = torch.from_numpy(episode_starts).bool().to(device)
        
        if action_masks is not None:
            action_masks_t = torch.from_numpy(action_masks).bool().to(device)
        else:
            action_masks_t = None
        
        # Forward
        with torch.no_grad():
            actions, _, _, new_states = self.forward(
                obs_t, lstm_states_t, episode_starts_t, 
                action_masks_t, deterministic=deterministic
            )
        
        # Convert back
        actions_np = actions.cpu().numpy()
        new_states_np = (
            new_states[0].cpu().numpy(),
            new_states[1].cpu().numpy(),
        )
        
        if was_1d:
            actions_np = actions_np[0]
        
        return actions_np, new_states_np
    
    def batch_predict(
        self,
        obs_batch: np.ndarray,
        action_masks_batch: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch inference per MCTS (senza hidden states).
        
        Args:
            obs_batch: [N, obs_dim]
            action_masks_batch: [N, n_actions]
            
        Returns:
            (priors, values): [N, n_actions] e [N]
        """
        self.eval()
        
        n = obs_batch.shape[0]
        device = next(self.parameters()).device
        
        # Convert
        obs_t = torch.from_numpy(obs_batch).float().to(device)
        
        # Use zero hidden states (stateless for MCTS)
        lstm_states = self.get_initial_state(n)
        episode_starts = torch.zeros(n, dtype=torch.bool, device=device)
        
        with torch.no_grad():
            features, _ = self.features_extractor(obs_t, lstm_states, episode_starts)
            action_logits = self.action_net(features)
            
            if action_masks_batch is not None:
                masks_t = torch.from_numpy(action_masks_batch).bool().to(device)
                action_logits = action_logits.masked_fill(~masks_t, float('-inf'))
            
            priors = F.softmax(action_logits, dim=-1).cpu().numpy()
            values = self.value_net(features).squeeze(-1).cpu().numpy()
        
        return priors, values


# Alias per compatibilità
MaskableRecurrentPolicy = MaskableRecurrentActorCriticPolicy


if __name__ == "__main__":
    print("Testing MaskableRecurrentPolicy...")
    
    # Create policy
    obs_space = gym.spaces.Box(low=0, high=1, shape=(OBSERVATION_DIM,), dtype=np.float32)
    act_space = gym.spaces.Discrete(ACTION_DIM)
    
    policy = MaskableRecurrentPolicy(obs_space, act_space)
    print(f"Policy created: {sum(p.numel() for p in policy.parameters())} parameters")
    
    # Test forward
    batch_size = 4
    obs = torch.randn(batch_size, OBSERVATION_DIM)
    lstm_states = policy.get_initial_state(batch_size)
    episode_starts = torch.zeros(batch_size, dtype=torch.bool)
    action_masks = torch.ones(batch_size, ACTION_DIM, dtype=torch.bool)
    action_masks[:, 10:] = False  # Solo prime 10 azioni valide
    
    actions, values, log_probs, new_states = policy.forward(
        obs, lstm_states, episode_starts, action_masks
    )
    
    print(f"Actions: {actions.shape}, Values: {values.shape}")
    print(f"Hidden states: h={new_states[0].shape}, c={new_states[1].shape}")
    
    # Test evaluate
    values2, log_probs2, entropy = policy.evaluate_actions(
        obs, actions, lstm_states, episode_starts, action_masks
    )
    print(f"Evaluate: values={values2.shape}, log_probs={log_probs2.shape}, entropy={entropy.mean():.3f}")
    
    # Test predict (numpy)
    obs_np = np.random.randn(OBSERVATION_DIM).astype(np.float32)
    masks_np = np.ones(ACTION_DIM, dtype=bool)
    action, new_states_np = policy.predict(obs_np, action_masks=masks_np)
    print(f"Predict: action={action}, states shapes={new_states_np[0].shape}")
    
    # Test batch_predict
    obs_batch = np.random.randn(8, OBSERVATION_DIM).astype(np.float32)
    priors, vals = policy.batch_predict(obs_batch)
    print(f"Batch predict: priors={priors.shape}, values={vals.shape}")
    
    print("✅ MaskableRecurrentPolicy OK!")
