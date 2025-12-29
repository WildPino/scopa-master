"""
RecurrentRolloutBuffer - Buffer per training PPO ricorrente.

Gestisce:
- Memorizzazione esperienze con hidden states LSTM
- Reset degli hidden states su episode boundaries
- Generazione di minibatch per training sequenziale

Questo è il secondo componente CRITICO per training ricorrente.
"""
from __future__ import annotations

from typing import Generator, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.vec_env import VecNormalize


class RecurrentRolloutBufferSamples(NamedTuple):
    """Batch di samples per training."""
    observations: Tensor
    actions: Tensor
    old_values: Tensor
    old_log_probs: Tensor
    advantages: Tensor
    returns: Tensor
    lstm_states: Tuple[Tensor, Tensor]
    episode_starts: Tensor
    action_masks: Tensor


class RecurrentRolloutBuffer(BaseBuffer):
    """
    Rollout buffer per policy ricorrenti con action masking.
    
    Mantiene:
    - observations, actions, rewards, dones per ogni step
    - hidden_states LSTM per ogni step
    - action_masks per ogni step
    - episode_starts per reset hidden states
    
    Layout hidden states: (n_layers, n_envs, hidden_size)
    Durante sampling: traspone a (n_envs, n_layers, hidden_size)
    
    Usage:
        buffer = RecurrentRolloutBuffer(...)
        
        # Durante rollout:
        buffer.add(obs, action, reward, done, value, log_prob, lstm_state, action_mask)
        
        # Dopo rollout:
        buffer.compute_returns_and_advantage(last_values, dones)
        
        # Training:
        for batch in buffer.get(batch_size):
            # train on batch
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        device: Union[torch.device, str] = "cpu",
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
        )
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        
        self.observations = None
        self.actions = None
        self.rewards = None
        self.returns = None
        self.advantages = None
        self.values = None
        self.log_probs = None
        self.episode_starts = None
        self.action_masks = None
        
        # Hidden states: (n_steps, n_layers, n_envs, hidden_size)
        self.lstm_states_h = None
        self.lstm_states_c = None
        
        self.generator_ready = False
        self.reset()
    
    def reset(self) -> None:
        """Reset del buffer per nuovo rollout."""
        obs_shape = self.observation_space.shape
        action_dim = self.action_space.n
        
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs) + obs_shape, 
            dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs), 
            dtype=np.int64
        )
        self.rewards = np.zeros(
            (self.buffer_size, self.n_envs), 
            dtype=np.float32
        )
        self.returns = np.zeros(
            (self.buffer_size, self.n_envs), 
            dtype=np.float32
        )
        self.advantages = np.zeros(
            (self.buffer_size, self.n_envs), 
            dtype=np.float32
        )
        self.values = np.zeros(
            (self.buffer_size, self.n_envs), 
            dtype=np.float32
        )
        self.log_probs = np.zeros(
            (self.buffer_size, self.n_envs), 
            dtype=np.float32
        )
        self.episode_starts = np.zeros(
            (self.buffer_size, self.n_envs), 
            dtype=np.float32
        )
        self.action_masks = np.ones(
            (self.buffer_size, self.n_envs, action_dim), 
            dtype=np.float32
        )
        
        # Hidden states
        self.lstm_states_h = np.zeros(
            (self.buffer_size, self.lstm_num_layers, self.n_envs, self.lstm_hidden_size),
            dtype=np.float32
        )
        self.lstm_states_c = np.zeros(
            (self.buffer_size, self.lstm_num_layers, self.n_envs, self.lstm_hidden_size),
            dtype=np.float32
        )
        
        self.pos = 0
        self.full = False
        self.generator_ready = False
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
        lstm_states: Tuple[np.ndarray, np.ndarray],
        action_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Aggiunge un timestep al buffer.
        
        Args:
            obs: [n_envs, obs_dim]
            action: [n_envs]
            reward: [n_envs]
            episode_start: [n_envs] True se episodio appena iniziato
            value: [n_envs]
            log_prob: [n_envs]
            lstm_states: (h_n, c_n) where each is [n_layers, n_envs, hidden]
            action_mask: [n_envs, n_actions] o None
        """
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1)
        
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = np.array(value).copy().flatten()
        self.log_probs[self.pos] = np.array(log_prob).copy()
        
        # LSTM states
        h_n, c_n = lstm_states
        self.lstm_states_h[self.pos] = np.array(h_n).copy()
        self.lstm_states_c[self.pos] = np.array(c_n).copy()
        
        # Action mask
        if action_mask is not None:
            self.action_masks[self.pos] = np.array(action_mask).copy()
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantage(
        self,
        last_values: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """
        Calcola GAE advantages e returns.
        
        Args:
            last_values: [n_envs] valori per ultimo stato
            dones: [n_envs] se episodio terminato
        """
        last_values = last_values.flatten()
        
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            
            delta = (
                self.rewards[step] 
                + self.gamma * next_values * next_non_terminal 
                - self.values[step]
            )
            last_gae_lam = (
                delta 
                + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        
        self.returns = self.advantages + self.values
        self.generator_ready = True
    
    def get(
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[RecurrentRolloutBufferSamples, None, None]:
        """
        Genera batch per training.
        
        Per policy ricorrenti, genera sequenze complete per ambiente
        invece di samples individuali shuffled.
        
        Args:
            batch_size: numero di ambienti per batch (None = tutti)
            
        Yields:
            RecurrentRolloutBufferSamples per ogni batch
        """
        assert self.generator_ready, "Chiama compute_returns_and_advantage prima"
        
        if batch_size is None:
            batch_size = self.n_envs
        
        # Per training ricorrente, iteriamo sulle sequenze complete per env
        # Non shuffliamo nel tempo per preservare dipendenze temporali
        
        indices = np.arange(self.n_envs)
        np.random.shuffle(indices)
        
        start_idx = 0
        while start_idx < self.n_envs:
            end_idx = min(start_idx + batch_size, self.n_envs)
            env_indices = indices[start_idx:end_idx]
            
            yield self._get_samples(env_indices)
            
            start_idx = end_idx
    
    def _get_samples(
        self,
        env_indices: np.ndarray,
    ) -> RecurrentRolloutBufferSamples:
        """
        Estrae samples per specifici ambienti.
        
        Args:
            env_indices: indici degli ambienti da estrarre
            
        Returns:
            RecurrentRolloutBufferSamples con shape [batch_size, seq_len, ...]
        """
        batch_size = len(env_indices)
        
        # Estrai dati: [buffer_size, n_envs, ...] -> [buffer_size, batch_size, ...]
        # Poi trasponi a [batch_size, buffer_size, ...] per training
        
        observations = self.observations[:, env_indices]  # [seq, batch, obs_dim]
        actions = self.actions[:, env_indices]  # [seq, batch]
        values = self.values[:, env_indices]  # [seq, batch]
        log_probs = self.log_probs[:, env_indices]  # [seq, batch]
        advantages = self.advantages[:, env_indices]  # [seq, batch]
        returns = self.returns[:, env_indices]  # [seq, batch]
        episode_starts = self.episode_starts[:, env_indices]  # [seq, batch]
        action_masks = self.action_masks[:, env_indices]  # [seq, batch, n_actions]
        
        # LSTM states: prendi solo il primo timestep per ogni sequenza
        # [buffer_size, n_layers, n_envs, hidden] -> [n_layers, batch, hidden]
        lstm_h = self.lstm_states_h[0, :, env_indices, :]  # [n_layers, batch, hidden]
        lstm_c = self.lstm_states_c[0, :, env_indices, :]
        
        # Normalizza advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Trasponi a [batch, seq, ...]
        observations = observations.transpose(1, 0, 2)  # [batch, seq, obs_dim]
        actions = actions.transpose(1, 0)  # [batch, seq]
        values = values.transpose(1, 0)
        log_probs = log_probs.transpose(1, 0)
        advantages = advantages.transpose(1, 0)
        returns = returns.transpose(1, 0)
        episode_starts = episode_starts.transpose(1, 0)
        action_masks = action_masks.transpose(1, 0, 2)  # [batch, seq, n_actions]
        
        # Flatten per training: [batch * seq, ...]
        seq_len = self.buffer_size
        observations = observations.reshape(batch_size * seq_len, -1)
        actions = actions.reshape(-1)
        values = values.reshape(-1)
        log_probs = log_probs.reshape(-1)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)
        episode_starts = episode_starts.reshape(-1)
        action_masks = action_masks.reshape(batch_size * seq_len, -1)
        
        return RecurrentRolloutBufferSamples(
            observations=self.to_torch(observations),
            actions=self.to_torch(actions).long(),
            old_values=self.to_torch(values),
            old_log_probs=self.to_torch(log_probs),
            advantages=self.to_torch(advantages),
            returns=self.to_torch(returns),
            lstm_states=(self.to_torch(lstm_h), self.to_torch(lstm_c)),
            episode_starts=self.to_torch(episode_starts).bool(),
            action_masks=self.to_torch(action_masks).bool(),
        )
    
    def to_torch(self, array: np.ndarray) -> Tensor:
        """Converte numpy array in torch tensor."""
        return torch.as_tensor(array, device=self.device)


if __name__ == "__main__":
    print("Testing RecurrentRolloutBuffer...")
    
    import gymnasium as gym
    
    obs_space = gym.spaces.Box(low=0, high=1, shape=(296,), dtype=np.float32)
    act_space = gym.spaces.Discrete(40)
    
    buffer = RecurrentRolloutBuffer(
        buffer_size=64,
        observation_space=obs_space,
        action_space=act_space,
        lstm_hidden_size=256,
        lstm_num_layers=2,
        n_envs=4,
    )
    
    print(f"Buffer created: {buffer.buffer_size} steps × {buffer.n_envs} envs")
    
    # Simula rollout
    for step in range(64):
        obs = np.random.randn(4, 296).astype(np.float32)
        actions = np.random.randint(0, 40, size=4)
        rewards = np.random.randn(4).astype(np.float32)
        episode_starts = np.zeros(4, dtype=np.float32)
        if step % 20 == 0:
            episode_starts[0] = 1.0  # Simula reset
        values = np.random.randn(4).astype(np.float32)
        log_probs = np.random.randn(4).astype(np.float32)
        lstm_h = np.random.randn(2, 4, 256).astype(np.float32)
        lstm_c = np.random.randn(2, 4, 256).astype(np.float32)
        masks = np.ones((4, 40), dtype=np.float32)
        
        buffer.add(
            obs, actions, rewards, episode_starts,
            values, log_probs, (lstm_h, lstm_c), masks
        )
    
    print(f"Buffer filled: pos={buffer.pos}, full={buffer.full}")
    
    # Compute GAE
    last_values = np.random.randn(4).astype(np.float32)
    dones = np.zeros(4)
    buffer.compute_returns_and_advantage(last_values, dones)
    print("GAE computed")
    
    # Test generator
    batches = list(buffer.get(batch_size=2))
    print(f"Generated {len(batches)} batches")
    
    batch = batches[0]
    print(f"Batch shapes:")
    print(f"  obs: {batch.observations.shape}")
    print(f"  actions: {batch.actions.shape}")
    print(f"  advantages: {batch.advantages.shape}")
    print(f"  lstm_h: {batch.lstm_states[0].shape}")
    print(f"  action_masks: {batch.action_masks.shape}")
    
    print("✅ RecurrentRolloutBuffer OK!")
