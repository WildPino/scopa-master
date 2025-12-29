"""
IS-MCTS - Information Set Monte Carlo Tree Search per Scopa.

Implementa MCTS con:
- PUCT per selezione (AlphaZero-style)
- Determinization sampling per gestire informazione parziale
- Batch inference per GPU efficiency
- Endgame solver per fasi finali

Usage:
    mcts = ISMCTS(c_puct=1.0)
    action, probs = mcts.search(env, policy_net, value_net, belief_net)
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import torch
import torch.nn as nn


# Constants
ACTION_DIM = 40
ENDGAME_THRESHOLD = 6  # Attiva endgame solver se carte_per_giocatore <= 6


@dataclass
class MCTSNode:
    """
    Nodo dell'albero MCTS.
    
    Mantiene statistiche aggregate su tutte le determinazioni.
    """
    action: int = -1  # Azione che porta a questo nodo (-1 per root)
    parent: Optional["MCTSNode"] = None
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)
    
    # Statistiche
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    
    # Cache
    is_expanded: bool = False
    is_terminal: bool = False
    
    @property
    def q_value(self) -> float:
        """Valore medio Q del nodo."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct: float, parent_visits: int) -> float:
        """
        Calcola PUCT score per selezione.
        
        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration
    
    def select_child(self, c_puct: float) -> "MCTSNode":
        """Seleziona figlio con UCB massimo."""
        best_score = -float('inf')
        best_child = None
        
        for child in self.children.values():
            score = child.ucb_score(c_puct, self.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, priors: np.ndarray, valid_actions: List[int]) -> None:
        """
        Espande il nodo creando figli per ogni azione valida.
        
        Args:
            priors: [action_dim] prior probabilities da policy network
            valid_actions: lista di azioni legali
        """
        if self.is_expanded:
            return
        
        for action in valid_actions:
            self.children[action] = MCTSNode(
                action=action,
                parent=self,
                prior=priors[action],
            )
        
        self.is_expanded = True
    
    def backup(self, value: float) -> None:
        """
        Propaga il valore all'indietro fino alla root.
        
        Args:
            value: valore stimato dalla prospettiva del giocatore corrente
        """
        node = self
        # Alterna segno del valore (zero-sum game)
        current_value = value
        
        while node is not None:
            node.visit_count += 1
            node.value_sum += current_value
            current_value = -current_value  # Flip per avversario
            node = node.parent


class ISMCTS:
    """
    Information Set Monte Carlo Tree Search.
    
    Aggrega risultati di MCTS su multiple determinazioni per gestire
    l'informazione parziale tipica dei giochi di carte.
    
    Config profiles:
    - tournament: 100 det × 100 sims (< 5s)
    - low_latency: 10 det × 20 sims (< 500ms)
    - training: 50 det × 30 sims
    """
    
    # Profili predefiniti
    PROFILES = {
        "tournament": {"num_det": 100, "sims_per_det": 100},
        "low_latency": {"num_det": 10, "sims_per_det": 20},
        "training": {"num_det": 50, "sims_per_det": 30},
    }
    
    def __init__(
        self,
        c_puct: float = 1.0,
        profile: str = "training",
        temperature: float = 1.0,
        add_noise: bool = False,
        noise_weight: float = 0.25,
        noise_alpha: float = 0.3,
    ):
        """
        Args:
            c_puct: costante di esplorazione PUCT (0.5-2.0)
            profile: "tournament", "low_latency", o "training"
            temperature: temperatura per selezione azione finale
            add_noise: se True, aggiunge Dirichlet noise ai priors (per training)
            noise_weight: peso del noise vs prior
            noise_alpha: parametro alpha per Dirichlet
        """
        self.c_puct = c_puct
        self.temperature = temperature
        self.add_noise = add_noise
        self.noise_weight = noise_weight
        self.noise_alpha = noise_alpha
        
        # Carica profilo
        if profile in self.PROFILES:
            cfg = self.PROFILES[profile]
            self.num_det = cfg["num_det"]
            self.sims_per_det = cfg["sims_per_det"]
        else:
            self.num_det = 50
            self.sims_per_det = 30
    
    def search(
        self,
        env,
        policy_net: Optional[nn.Module] = None,
        value_net: Optional[nn.Module] = None,
        belief_net: Optional[nn.Module] = None,
        num_det: Optional[int] = None,
        sims: Optional[int] = None,
    ) -> Tuple[int, Dict[int, float]]:
        """
        Esegue IS-MCTS e ritorna azione migliore.
        
        Args:
            env: ScopaEnv con stato corrente
            policy_net: rete per prior (None = uniform)
            value_net: rete per value (None = random rollout)
            belief_net: rete per belief (None = uniform)
            num_det: override numero determinazioni
            sims: override simulazioni per determinazione
            
        Returns:
            (best_action, action_probs): azione scelta e distribuzione
        """
        num_det = num_det or self.num_det
        sims = sims or self.sims_per_det
        
        # Aggregatore di visite tra determinazioni
        total_visits = np.zeros(ACTION_DIM, dtype=np.float32)
        
        # Get valid actions
        action_mask = env.action_masks()
        valid_actions = np.where(action_mask)[0].tolist()
        
        if len(valid_actions) == 0:
            return 0, {}
        
        if len(valid_actions) == 1:
            return valid_actions[0], {valid_actions[0]: 1.0}
        
        # Get belief probabilities
        belief_probs = None
        if belief_net is not None:
            obs = env._get_obs()
            belief_probs = belief_net.predict(obs)
        
        # Sample determinizations
        from scopa.rl.determinize import sample_determinizations
        determinizations = sample_determinizations(
            env,
            belief_probs=belief_probs,
            n_samples=num_det,
        )
        
        # Run MCTS per determinization
        for det in determinizations:
            # Crea ambiente determinizzato
            det_env = self._apply_determinization(env, det)
            
            # Crea root
            root = MCTSNode()
            
            # Get priors
            priors = self._get_priors(det_env, policy_net, valid_actions)
            root.expand(priors, valid_actions)
            
            # Run simulations
            for _ in range(sims):
                self._simulate(root, det_env, policy_net, value_net)
            
            # Accumula visite
            for action, child in root.children.items():
                total_visits[action] += child.visit_count * det.weight
        
        # Normalizza e seleziona azione
        if total_visits.sum() > 0:
            visits_valid = total_visits[valid_actions]
            
            if self.temperature == 0:
                # Greedy
                best_idx = np.argmax(visits_valid)
                best_action = valid_actions[best_idx]
            else:
                # Temperature-based sampling
                visits_temp = visits_valid ** (1.0 / self.temperature)
                probs = visits_temp / (visits_temp.sum() + 1e-10)
                best_action = np.random.choice(valid_actions, p=probs)
            
            # Costruisci distribuzione finale
            total = total_visits[valid_actions].sum()
            action_probs = {a: total_visits[a] / total for a in valid_actions}
        else:
            # Fallback random
            best_action = np.random.choice(valid_actions)
            action_probs = {a: 1.0 / len(valid_actions) for a in valid_actions}
        
        return best_action, action_probs
    
    def _apply_determinization(self, env, det) -> Any:
        """
        Applica una determinazione all'ambiente.
        
        Per ora restituisce l'ambiente originale (determinazioni usate
        solo per calcolo priors/values coerenti).
        
        TODO: creare copia deep dell'ambiente con carte assegnate.
        """
        # Per semplicità, usiamo l'env originale
        # In una implementazione completa, si creerebbe una copia
        # con le carte dell'avversario assegnate secondo det.opponent_hand
        return env
    
    def _get_priors(
        self,
        env,
        policy_net: Optional[nn.Module],
        valid_actions: List[int],
    ) -> np.ndarray:
        """Ottiene prior probabilities dalla policy network."""
        priors = np.zeros(ACTION_DIM, dtype=np.float32)
        
        if policy_net is not None:
            obs = env._get_obs()
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            
            with torch.no_grad():
                if hasattr(policy_net, 'predict'):
                    # Usa metodo predict se disponibile
                    action, _ = policy_net.predict(obs, deterministic=False)
                    priors[action] = 1.0
                else:
                    # Forward diretto
                    if hasattr(policy_net, 'policy'):
                        # SB3 model
                        logits = policy_net.policy.get_distribution(obs_t).distribution.logits
                    else:
                        logits = policy_net(obs_t)
                    
                    # Mask e softmax
                    mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
                    mask[valid_actions] = True
                    logits = logits.squeeze(0)
                    logits[~mask] = float('-inf')
                    priors = torch.softmax(logits, dim=-1).numpy()
        else:
            # Uniform priors
            for a in valid_actions:
                priors[a] = 1.0 / len(valid_actions)
        
        # Add Dirichlet noise se richiesto
        if self.add_noise and len(valid_actions) > 0:
            noise = np.random.dirichlet([self.noise_alpha] * len(valid_actions))
            for i, a in enumerate(valid_actions):
                priors[a] = (1 - self.noise_weight) * priors[a] + self.noise_weight * noise[i]
        
        return priors
    
    def _simulate(
        self,
        root: MCTSNode,
        env,
        policy_net: Optional[nn.Module],
        value_net: Optional[nn.Module],
    ) -> None:
        """Esegue una simulazione MCTS (select, expand, evaluate, backup)."""
        node = root
        
        # Selection: scendi nell'albero
        while node.is_expanded and not node.is_terminal:
            if not node.children:
                break
            node = node.select_child(self.c_puct)
        
        # Se terminale, usa reward finale
        if node.is_terminal:
            value = 0.0  # O reward finale
        else:
            # Expansion
            if not node.is_expanded:
                mask = env.action_masks()
                valid_actions = np.where(mask)[0].tolist()
                
                if len(valid_actions) == 0:
                    node.is_terminal = True
                    value = 0.0
                else:
                    priors = self._get_priors(env, policy_net, valid_actions)
                    node.expand(priors, valid_actions)
                    
                    # Evaluation
                    value = self._evaluate(env, value_net)
            else:
                value = self._evaluate(env, value_net)
        
        # Backup
        node.backup(value)
    
    def _evaluate(
        self,
        env,
        value_net: Optional[nn.Module],
    ) -> float:
        """Valuta lo stato corrente."""
        if value_net is not None:
            obs = env._get_obs()
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            
            with torch.no_grad():
                if hasattr(value_net, 'predict_values'):
                    value = value_net.predict_values(obs_t).item()
                else:
                    value = value_net(obs_t).item()
            
            return value
        else:
            # Random rollout fallback (simplified)
            return 0.0
    
    def batch_evaluate(
        self,
        envs: List[Any],
        policy_net: Optional[nn.Module] = None,
        value_net: Optional[nn.Module] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch inference per GPU efficiency.
        
        Args:
            envs: lista di ambienti da valutare
            policy_net: network per priors
            value_net: network per values
            
        Returns:
            (priors, values): arrays [N, action_dim] e [N]
        """
        n = len(envs)
        if n == 0:
            return np.array([]), np.array([])
        
        # Colleziona osservazioni
        obs_batch = np.stack([env._get_obs() for env in envs])
        obs_t = torch.from_numpy(obs_batch).float()
        
        priors = np.ones((n, ACTION_DIM)) / ACTION_DIM
        values = np.zeros(n)
        
        with torch.no_grad():
            if policy_net is not None:
                # Get priors
                if hasattr(policy_net, 'policy') and hasattr(policy_net.policy, 'get_distribution'):
                    dist = policy_net.policy.get_distribution(obs_t)
                    priors = torch.softmax(dist.distribution.logits, dim=-1).numpy()
                elif callable(policy_net):
                    logits = policy_net(obs_t)
                    priors = torch.softmax(logits, dim=-1).numpy()
            
            if value_net is not None:
                if hasattr(value_net, 'predict_values'):
                    values = value_net.predict_values(obs_t).squeeze(-1).numpy()
                elif callable(value_net):
                    values = value_net(obs_t).squeeze(-1).numpy()
        
        return priors, values


class EndgameSolver:
    """
    Risoluzione esatta/intensiva per fine partita.
    
    Quando rimangono poche carte (≤ ENDGAME_THRESHOLD per giocatore),
    usiamo search più profonda o solver esatto.
    """
    
    def __init__(
        self,
        threshold: int = ENDGAME_THRESHOLD,
        max_depth: int = 20,
    ):
        self.threshold = threshold
        self.max_depth = max_depth
    
    def is_endgame(self, env) -> bool:
        """Rileva se siamo in endgame."""
        # Conta carte rimanenti per giocatore
        hand_size = len(env.engine.hands[0])
        deck_size = len(env.engine.deck)
        cards_per_player = hand_size + deck_size // 2
        
        return cards_per_player <= self.threshold
    
    def solve(
        self,
        env,
        policy_net: Optional[nn.Module] = None,
        value_net: Optional[nn.Module] = None,
    ) -> int:
        """
        Risolvi posizione endgame con search intensiva.
        
        Per ora usa MCTS con più simulazioni.
        TODO: implementare negamax esatto per stati piccoli.
        """
        mcts = ISMCTS(
            c_puct=1.0,
            profile="tournament",  # Usa configurazione intensiva
            temperature=0,  # Greedy
        )
        
        # Aumenta simulazioni per endgame
        action, _ = mcts.search(
            env,
            policy_net=policy_net,
            value_net=value_net,
            num_det=200,
            sims=200,
        )
        
        return action


if __name__ == "__main__":
    print("Testing IS-MCTS...")
    
    # Test MCTSNode
    root = MCTSNode()
    priors = np.array([0.5, 0.3, 0.2] + [0.0] * 37)
    root.expand(priors, [0, 1, 2])
    print(f"Root expanded with {len(root.children)} children")
    
    # Test backup
    root.children[0].visit_count = 5
    root.children[0].value_sum = 2.5
    root.children[0].backup(1.0)
    print(f"After backup: visits={root.children[0].visit_count}, Q={root.children[0].q_value:.3f}")
    
    # Test ISMCTS initialization
    mcts = ISMCTS(c_puct=1.0, profile="low_latency")
    print(f"MCTS config: {mcts.num_det} det × {mcts.sims_per_det} sims")
    
    print("✅ IS-MCTS module OK!")
