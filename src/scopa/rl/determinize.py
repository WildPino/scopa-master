"""
Determinization - Sampling di deal coerenti per IS-MCTS.

Genera N deal (stati "determinizzati") coerenti con:
- Carte visibili (mano AI, tavolo, catture)
- Storia delle mosse giocate
- Belief probabilities da BeliefNet

Include validatore rigoroso per evitare stati incoerenti.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np


NUM_CARDS = 40


@dataclass
class DeterminizedState:
    """
    Stato determinizzato con assegnazione completa delle carte nascoste.
    
    Contiene una copia dello stato di gioco con:
    - Mano avversario assegnata completamente
    - Mazzo residuo assegnato
    """
    opponent_hand: List[int]  # Indici carte in mano all'avversario
    deck: List[int]           # Indici carte rimanenti nel mazzo
    weight: float = 1.0       # Peso per importance sampling
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "opponent_hand": self.opponent_hand,
            "deck": self.deck,
            "weight": self.weight,
        }


def get_visible_cards(env) -> Set[int]:
    """
    Estrae gli indici di tutte le carte visibili all'AI.
    
    Include:
    - Carte in mano all'AI
    - Carte sul tavolo
    - Carte catturate da entrambi
    - Carte nella history (giocate durante la partita)
    """
    visible = set()
    
    # Mano AI
    for card in env.engine.hands[0]:
        visible.add(card.index)
    
    # Tavolo
    for card in env.engine.table:
        visible.add(card.index)
    
    # Catture AI
    for card in env.engine.captured[0]:
        visible.add(card.index)
    
    # Catture avversario
    for card in env.engine.captured[1]:
        visible.add(card.index)
    
    # History buffer (carte giocate)
    if hasattr(env, 'history'):
        for i, val in enumerate(env.history):
            if val > 0.5:  # Carta è stata giocata
                visible.add(i)
    
    return visible


def get_hidden_cards(env) -> List[int]:
    """
    Ritorna gli indici delle carte non visibili all'AI.
    
    Queste possono essere in mano all'avversario o nel mazzo.
    """
    visible = get_visible_cards(env)
    return [i for i in range(NUM_CARDS) if i not in visible]


def validate_determinization(
    det_state: DeterminizedState,
    env,
    visible_cards: Set[int]
) -> Tuple[bool, str]:
    """
    Validatore rigoroso per determinization.
    
    Verifica:
    1. Nessuna carta duplicata (O(1) con bitmask)
    2. Carte visibili non riassegnate
    3. Numero corretto di carte per mano/mazzo
    
    Args:
        det_state: stato determinizzato da validare
        env: ambiente originale
        visible_cards: set di carte visibili
        
    Returns:
        (is_valid, error_message)
    """
    # 1. Check duplicati usando bitmask
    seen_mask = 0
    all_cards = det_state.opponent_hand + det_state.deck
    
    for card_idx in all_cards:
        if card_idx < 0 or card_idx >= NUM_CARDS:
            return False, f"Card index {card_idx} out of range"
        
        bit = 1 << card_idx
        if seen_mask & bit:
            return False, f"Duplicate card {card_idx}"
        seen_mask |= bit
    
    # 2. Check che nessuna carta visibile sia stata riassegnata
    for card_idx in all_cards:
        if card_idx in visible_cards:
            return False, f"Visible card {card_idx} reassigned"
    
    # 3. Check numero carte
    expected_opp_hand = len(env.engine.hands[1])
    if len(det_state.opponent_hand) != expected_opp_hand:
        return False, f"Wrong opponent hand size: {len(det_state.opponent_hand)} vs {expected_opp_hand}"
    
    expected_deck = len(env.engine.deck)
    if len(det_state.deck) != expected_deck:
        return False, f"Wrong deck size: {len(det_state.deck)} vs {expected_deck}"
    
    return True, "OK"


def sample_determinizations(
    env,
    belief_probs: Optional[np.ndarray] = None,
    n_samples: int = 50,
    validate: bool = True,
    max_attempts_per_sample: int = 10,
) -> List[DeterminizedState]:
    """
    Genera N deal coerenti con storico e belief.
    
    Usa weighted sampling basato sulle probabilità di BeliefNet.
    Fallback a uniform sampling se belief_probs non fornito.
    
    Args:
        env: ScopaEnv con stato corrente
        belief_probs: [40] probabilità per carta di essere in mano avversario
        n_samples: numero di determinazioni da generare
        validate: se True, valida ogni determinization
        max_attempts_per_sample: tentativi massimi per sample valido
        
    Returns:
        Lista di DeterminizedState
    """
    visible_cards = get_visible_cards(env)
    hidden_cards = get_hidden_cards(env)
    
    if len(hidden_cards) == 0:
        # Nessuna carta nascosta - stato già determinato
        return [DeterminizedState(
            opponent_hand=[c.index for c in env.engine.hands[1]],
            deck=[c.index for c in env.engine.deck],
            weight=1.0
        )]
    
    # Numero di carte da assegnare
    num_opp_hand = len(env.engine.hands[1])
    num_deck = len(env.engine.deck)
    total_hidden = num_opp_hand + num_deck
    
    assert len(hidden_cards) == total_hidden, \
        f"Hidden cards mismatch: {len(hidden_cards)} vs {total_hidden}"
    
    # Costruisci probabilità per carte nascoste
    if belief_probs is not None:
        hidden_probs = np.array([belief_probs[i] for i in hidden_cards])
        # Normalizza
        hidden_probs = hidden_probs / (hidden_probs.sum() + 1e-10)
    else:
        # Uniform
        hidden_probs = np.ones(len(hidden_cards)) / len(hidden_cards)
    
    samples = []
    
    for _ in range(n_samples):
        valid_sample = False
        
        for attempt in range(max_attempts_per_sample):
            # Sample carte per mano avversario (senza replacement)
            if num_opp_hand > 0 and num_opp_hand <= len(hidden_cards):
                try:
                    # Weighted sampling senza replacement
                    sampled_indices = np.random.choice(
                        len(hidden_cards),
                        size=num_opp_hand,
                        replace=False,
                        p=hidden_probs if np.all(hidden_probs > 0) else None
                    )
                    opp_hand = [hidden_cards[i] for i in sampled_indices]
                    
                    # Resto va nel mazzo
                    deck = [hidden_cards[i] for i in range(len(hidden_cards)) 
                            if i not in sampled_indices]
                except ValueError:
                    # Fallback to uniform if weighted fails
                    np.random.shuffle(hidden_cards)
                    opp_hand = hidden_cards[:num_opp_hand]
                    deck = hidden_cards[num_opp_hand:]
            else:
                opp_hand = []
                deck = hidden_cards[:]
            
            # Calcola peso per importance sampling
            if belief_probs is not None:
                weight = np.prod([belief_probs[c] for c in opp_hand]) + 1e-10
            else:
                weight = 1.0
            
            det_state = DeterminizedState(
                opponent_hand=list(opp_hand),
                deck=list(deck),
                weight=weight,
            )
            
            if validate:
                is_valid, error = validate_determinization(det_state, env, visible_cards)
                if is_valid:
                    valid_sample = True
                    break
            else:
                valid_sample = True
                break
        
        if valid_sample:
            samples.append(det_state)
    
    # Normalizza pesi
    if samples:
        total_weight = sum(s.weight for s in samples)
        for s in samples:
            s.weight /= total_weight
    
    return samples


class ParticleFilter:
    """
    Mantiene un insieme di particelle (ipotesi sullo stato nascosto).
    
    Ogni particella rappresenta una possibile assegnazione delle carte.
    I pesi vengono aggiornati quando arrivano nuove informazioni (mosse).
    """
    
    def __init__(
        self,
        n_particles: int = 200,
        resample_threshold: float = 0.5,
    ):
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        self.particles: List[DeterminizedState] = []
    
    def initialize(self, env, belief_probs: Optional[np.ndarray] = None) -> None:
        """Inizializza particelle dallo stato corrente."""
        self.particles = sample_determinizations(
            env,
            belief_probs=belief_probs,
            n_samples=self.n_particles,
            validate=True,
        )
    
    def update(
        self,
        action: int,
        observation: Any,
        likelihood_fn: Optional[callable] = None,
    ) -> None:
        """
        Aggiorna pesi delle particelle dopo un'osservazione.
        
        Args:
            action: azione eseguita
            observation: nuova osservazione
            likelihood_fn: funzione che calcola p(observation | particle)
        """
        if not self.particles:
            return
        
        if likelihood_fn is not None:
            # Aggiorna pesi based on likelihood
            for p in self.particles:
                p.weight *= likelihood_fn(p, action, observation)
        
        # Normalizza pesi
        total = sum(p.weight for p in self.particles)
        if total > 0:
            for p in self.particles:
                p.weight /= total
        
        # Resample se effective sample size troppo basso
        ess = self._effective_sample_size()
        if ess < self.n_particles * self.resample_threshold:
            self._resample()
    
    def _effective_sample_size(self) -> float:
        """Calcola effective sample size."""
        if not self.particles:
            return 0.0
        weights = [p.weight for p in self.particles]
        return 1.0 / (sum(w**2 for w in weights) + 1e-10)
    
    def _resample(self) -> None:
        """Resample particelle con replacement basato sui pesi."""
        if not self.particles:
            return
        
        weights = np.array([p.weight for p in self.particles])
        indices = np.random.choice(
            len(self.particles),
            size=self.n_particles,
            replace=True,
            p=weights,
        )
        
        # Crea nuove particelle con pesi uniformi
        new_particles = []
        for i in indices:
            p = self.particles[i]
            new_particles.append(DeterminizedState(
                opponent_hand=p.opponent_hand[:],
                deck=p.deck[:],
                weight=1.0 / self.n_particles,
            ))
        
        self.particles = new_particles
    
    def get_samples(self, n: int = 50) -> List[DeterminizedState]:
        """Ritorna n samples dalle particelle attuali."""
        if not self.particles:
            return []
        
        if n >= len(self.particles):
            return self.particles[:]
        
        weights = np.array([p.weight for p in self.particles])
        indices = np.random.choice(
            len(self.particles),
            size=n,
            replace=False,
            p=weights,
        )
        
        return [self.particles[i] for i in indices]


if __name__ == "__main__":
    # Test
    import sys
    sys.path.insert(0, str(__file__.rsplit('\\', 3)[0] if '\\' in __file__ else __file__.rsplit('/', 3)[0]))
    
    from scopa.rl import ScopaEnv
    
    print("Testing determinization...")
    env = ScopaEnv()
    env.reset()
    
    # Test con uniform belief
    samples = sample_determinizations(env, belief_probs=None, n_samples=10)
    print(f"Generated {len(samples)} samples with uniform belief")
    
    # Test con belief random
    belief = np.random.rand(NUM_CARDS)
    samples = sample_determinizations(env, belief_probs=belief, n_samples=10)
    print(f"Generated {len(samples)} samples with random belief")
    
    # Valida tutti i samples
    visible = get_visible_cards(env)
    for i, s in enumerate(samples):
        valid, msg = validate_determinization(s, env, visible)
        assert valid, f"Sample {i} invalid: {msg}"
    
    print("✅ All determinizations valid!")
    
    # Test particle filter
    pf = ParticleFilter(n_particles=50)
    pf.initialize(env, belief_probs=belief)
    print(f"ParticleFilter initialized with {len(pf.particles)} particles")
    print(f"ESS: {pf._effective_sample_size():.1f}")
    
    print("✅ Determinization OK!")
