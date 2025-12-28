"""
Scopa Environment - Gymnasium-compatible RL environment.

Sistema a Due Fasi:
- Fase 0: Selezione carta da giocare (40 azioni, mask = mano)
- Fase 1: Selezione carte da prendere (iterativa, auto-conferma quando somma = target)

Observation Space: 296 elementi
  [000-239] 240: Mano, Tavolo, Prese AI, Prese Avv, Opp Played Hand, Selected Fase 1
  [240]      1: Carte rimanenti nel mazzo (0.0 - 1.0)
  [241-251] 11: Statistiche normalizzate
  [252]      1: Last capture player (0/1)
  [253]      1: Capture mode (0 = Fase 0, 1 = Fase 1)
  [254]      1: Card played index (n/39.0)
  [255-294] 40: History buffer (cronologia carte giocate da mano)
  [295]      1: Starter player (0.0 = AI, 1.0 = Avv)

Action Space: 40 azioni (0-39 = indice carta)
"""
from __future__ import annotations

import logging
import random
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from scopa.game import Card, Suit, ScopaEngine
from scopa.config import OBSERVATION_DIM, ACTION_DIM, HISTORY_BUFFER_SIZE

# Type aliases
ObsType = np.ndarray
ActionType = int
InfoDict = Dict[str, Any]

logger = logging.getLogger(__name__)


class ScopaEnv(gym.Env):
    """
    Ambiente Scopa con supporto per diverse modalità di avversario.
    
    Args:
        opponent_mode: Modalità avversario
            - 'random': Mosse casuali (default)
            - 'self': Self-play con stesso modello
            - 'heuristic': Avversario euristico
            - 'mixed': Alterna casualmente
        model: Modello per self-play (opzionale)
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        opponent_mode: str = "random",
        model: Optional[Any] = None,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.engine = ScopaEngine()
        self.opponent_mode = opponent_mode
        self.model = model
        self.render_mode = render_mode
        
        # Action space: 40 carte
        self.action_space = spaces.Discrete(ACTION_DIM)
        
        # Observation space: 255 features normalizzate
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBSERVATION_DIM,), dtype=np.float32
        )
        
        # Stato interno fasi
        self.phase: int = 0  # 0 = selezione carta, 1 = selezione presa
        self.card_played: Optional[Card] = None
        self.selected_captures: List[Card] = []
        self.valid_capture_options: List[List[Card]] = []
        
        # Maschere tracking
        self._captured_masks = [np.zeros(40, dtype=np.int8) for _ in range(2)]
        self._opponent_played_this_hand = np.zeros(40, dtype=np.int8)
        self._ai_played_this_hand = np.zeros(40, dtype=np.int8)
        
        # History buffer per inferenza dinamica
        self.history = np.zeros(HISTORY_BUFFER_SIZE, dtype=np.float32)
        self.history_idx = 0
        self.starter_player = 0  # 0 = AI inizia, 1 = Avversario inizia
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[ObsType, InfoDict]:
        """Reset dell'ambiente per un nuovo episodio."""
        super().reset(seed=seed)
        
        self.engine.reset()
        self.phase = 0
        self.card_played = None
        self.selected_captures = []
        self.valid_capture_options = []
        
        self._captured_masks = [np.zeros(40, dtype=np.int8) for _ in range(2)]
        self._opponent_played_this_hand = np.zeros(40, dtype=np.int8)
        self._ai_played_this_hand = np.zeros(40, dtype=np.int8)
        
        # Reset history buffer
        self.history = np.zeros(HISTORY_BUFFER_SIZE, dtype=np.float32)
        self.history_idx = 0
        
        # 50% chance che l'avversario inizi
        info: InfoDict = {}
        if random.random() < 0.5:
            self.starter_player = 1  # Avversario inizia
            opp_move = self._do_opponent_turn()
            info["opponent_move"] = opp_move
        else:
            self.starter_player = 0  # AI inizia
        
        return self._get_obs(), info
    
    def step(self, action: ActionType) -> Tuple[ObsType, float, bool, bool, InfoDict]:
        """Esegue un'azione nell'ambiente."""
        if self.phase == 0:
            return self._step_phase0(action)
        else:
            return self._step_phase1(action)
    
    def action_masks(self) -> np.ndarray:
        """Ritorna la maschera delle azioni legali (per MaskablePPO)."""
        mask = np.zeros(40, dtype=bool)
        
        if self.phase == 0:
            # Fase 0: carte in mano
            for card in self.engine.hands[0]:
                mask[card.index] = True
        else:
            # Fase 1: carte dal tavolo che portano a presa valida
            current_indices = {c.index for c in self.selected_captures}
            current_sum = sum(c.rank for c in self.selected_captures)
            needed = self.card_played.rank - current_sum
            
            valid_next = set()
            for combo in self.valid_capture_options:
                combo_indices = {c.index for c in combo}
                if current_indices.issubset(combo_indices):
                    remaining = combo_indices - current_indices
                    for card in combo:
                        if card.index in remaining and card.rank <= needed:
                            valid_next.add(card.index)
            
            for idx in valid_next:
                mask[idx] = True
        
        return mask
    
    def set_model(self, model: Any) -> None:
        """Imposta il modello per self-play."""
        self.model = model
    
    def update_model_weights(self, state_dict: Dict[str, Any]) -> None:
        """
        Aggiorna i pesi della policy del modello locale.
        
        Questa funzione è compatibile con SubprocVecEnv perché riceve
        solo lo state_dict (tensori serializzabili) invece del modello intero.
        
        Se il modello non esiste, viene creato lazily alla prima chiamata.
        
        Args:
            state_dict: Dizionario con i pesi della policy
        """
        if self.model is None:
            # Crea modello locale per self-play (lazy initialization)
            self._create_local_model()
        
        if self.model is not None and hasattr(self.model, 'policy'):
            self.model.policy.load_state_dict(state_dict)
    
    def _create_local_model(self) -> None:
        """Crea un modello locale per self-play in ambienti subprocess."""
        try:
            from sb3_contrib import MaskablePPO
            from scopa.config import NETWORK_ARCH
            
            # Crea un ambiente dummy per inizializzare il modello
            policy_kwargs = dict(
                net_arch=dict(pi=NETWORK_ARCH["pi"], vf=NETWORK_ARCH["vf"])
            )
            
            self.model = MaskablePPO(
                "MlpPolicy",
                self,  # Usa self come environment
                verbose=0,
                device="cpu",  # Sempre CPU per inferenza locale
                policy_kwargs=policy_kwargs,
            )
        except Exception as e:
            logger.warning(f"Impossibile creare modello locale: {e}")
            self.model = None
    
    def render(self) -> Optional[str]:
        """Render testuale dello stato."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None
    
    # ========== PRIVATE METHODS ==========
    
    def _get_obs(self) -> ObsType:
        """Costruisce l'osservazione da 296 elementi."""
        # 1. Mano AI (40)
        hand_obs = np.zeros(40, dtype=np.float32)
        for card in self.engine.hands[0]:
            hand_obs[card.index] = 1.0
        
        # 2. Tavolo escluse carte selezionate (40)
        table_obs = np.zeros(40, dtype=np.float32)
        for card in self.engine.table:
            if card not in self.selected_captures:
                table_obs[card.index] = 1.0
        
        # 3-4. Carte catturate (40+40)
        captured_p0 = self._captured_masks[0].astype(np.float32)
        captured_p1 = self._captured_masks[1].astype(np.float32)
        
        # 5. Carte giocate dall'avversario questa mano (40)
        opponent_played = self._opponent_played_this_hand.astype(np.float32)
        
        # 6. Carte selezionate per presa (40)
        selected_obs = np.zeros(40, dtype=np.float32)
        for card in self.selected_captures:
            selected_obs[card.index] = 1.0
        
        # 7. Mazzo rimanente (1)
        deck_obs = np.array([len(self.engine.deck) / 40.0], dtype=np.float32)
        
        # 8. Statistiche (11)
        stats = np.array([
            self.engine.scope[0] / 10.0,
            len(self.engine.captured[0]) / 40.0,
            sum(1 for c in self.engine.captured[0] if c.suit == Suit.DENARI) / 10.0,
            1.0 if any(c.rank == 7 and c.suit == Suit.DENARI for c in self.engine.captured[0]) else 0.0,
            self.engine._get_primiera_score(self.engine.captured[0]) / 84.0,
            self.engine.scope[1] / 10.0,
            len(self.engine.captured[1]) / 40.0,
            sum(1 for c in self.engine.captured[1] if c.suit == Suit.DENARI) / 10.0,
            1.0 if any(c.rank == 7 and c.suit == Suit.DENARI for c in self.engine.captured[1]) else 0.0,
            self.engine._get_primiera_score(self.engine.captured[1]) / 84.0,
            len(self.engine.hands[1]) / 3.0,
        ], dtype=np.float32)
        
        # 9. Last capture player (1)
        last_capture = np.array([float(self.engine.last_capture_player)], dtype=np.float32)
        
        # 10. Capture mode (1)
        capture_mode = np.array([1.0 if self.phase == 1 else 0.0], dtype=np.float32)
        
        # 11. Card played index (1)
        card_idx = np.array([
            self.card_played.index / 39.0 if self.card_played else 0.0
        ], dtype=np.float32)
        
        # 12. History buffer (40) - cronologia carte giocate da mano
        history_obs = self.history.copy()
        
        # 13. Starter player (1) - chi ha iniziato la partita
        starter_obs = np.array([float(self.starter_player)], dtype=np.float32)
        
        return np.concatenate([
            hand_obs,        # 40  [0-39]
            table_obs,       # 40  [40-79]
            captured_p0,     # 40  [80-119]
            captured_p1,     # 40  [120-159]
            opponent_played, # 40  [160-199]
            selected_obs,    # 40  [200-239]
            deck_obs,        # 1   [240]
            stats,           # 11  [241-251]
            last_capture,    # 1   [252]
            capture_mode,    # 1   [253]
            card_idx,        # 1   [254]
            history_obs,     # 40  [255-294]
            starter_obs,     # 1   [295]
        ])  # Total: 296
    
    def _get_valid_captures(self, card: Card) -> List[List[Card]]:
        """Trova combinazioni valide di presa per una carta."""
        # Presa diretta obbligatoria
        direct = [c for c in self.engine.table if c.rank == card.rank]
        if direct:
            return [[c] for c in direct]
        
        # Combinazioni
        valid: List[List[Card]] = []
        for r in range(2, len(self.engine.table) + 1):
            for combo in combinations(self.engine.table, r):
                if sum(c.rank for c in combo) == card.rank:
                    valid.append(list(combo))
        
        return valid
    
    def _step_phase0(self, action: ActionType) -> Tuple[ObsType, float, bool, bool, InfoDict]:
        """Fase 0: Selezione carta da giocare."""
        # Trova carta
        card = next((c for c in self.engine.hands[0] if c.index == action), None)
        
        if card is None:
            logger.warning(f"FASE 0 - Azione invalida: {action}")
            return self._get_obs(), -10.0, True, False, {"invalid_action": True}
        
        self.card_played = card
        self.valid_capture_options = self._get_valid_captures(card)
        
        # Registra la carta nel history buffer (con safety check)
        if self.history_idx < HISTORY_BUFFER_SIZE:
            self.history[self.history_idx] = card.index / 39.0
            self.history_idx += 1
        
        # Traccia le carte giocate dall'AI questa mano (per self-play swap)
        self._ai_played_this_hand[card.index] = 1
        
        if len(self.valid_capture_options) == 0:
            # Calata sul tavolo
            self.engine.hands[0].remove(card)
            self.engine.table.append(card)
            
            ai_move = (card, [])
            opp_move = self._do_opponent_turn()
            self._check_deal_new_hand()
            
            return self._finish_step(ai_move, opp_move)
        
        elif len(self.valid_capture_options) == 1:
            # Una sola opzione: esegui automaticamente
            capture = self.valid_capture_options[0]
            self.engine.hands[0].remove(card)
            self._execute_capture(0, card, capture)
            
            ai_move = (card, capture)
            opp_move = self._do_opponent_turn()
            self._check_deal_new_hand()
            
            return self._finish_step(ai_move, opp_move)
        
        else:
            # Multiple opzioni: vai a Fase 1
            self.phase = 1
            self.selected_captures = []
            self.engine.hands[0].remove(card)
            
            return self._get_obs(), 0.0, False, False, {"phase": 1}
    
    def _step_phase1(self, action: ActionType) -> Tuple[ObsType, float, bool, bool, InfoDict]:
        """Fase 1: Selezione carte da prendere."""
        table_card = next(
            (c for c in self.engine.table if c.index == action and c not in self.selected_captures),
            None
        )
        
        if table_card is None:
            logger.warning(f"FASE 1 - Carta non valida: {action}")
            return self._get_obs(), -10.0, True, False, {"invalid_action": True}
        
        self.selected_captures.append(table_card)
        current_sum = sum(c.rank for c in self.selected_captures)
        
        if current_sum == self.card_played.rank:
            # Somma raggiunta: auto-conferma
            self._execute_capture(0, self.card_played, self.selected_captures)
            ai_move = (self.card_played, list(self.selected_captures))
            
            self.phase = 0
            self.card_played = None
            self.selected_captures = []
            
            opp_move = self._do_opponent_turn()
            self._check_deal_new_hand()
            
            return self._finish_step(ai_move, opp_move)
        
        elif current_sum > self.card_played.rank:
            logger.warning(f"FASE 1 - Somma superata: {current_sum} > {self.card_played.rank}")
            return self._get_obs(), -10.0, True, False, {"invalid_action": True}
        
        else:
            # Continua selezione
            return self._get_obs(), 0.0, False, False, {"phase": 1}
    
    def _execute_capture(self, player: int, card: Card, taken: List[Card]) -> None:
        """Esegue una presa e aggiorna maschere."""
        self.engine.captured[player].append(card)
        self.engine.captured[player].extend(taken)
        
        for c in taken:
            self.engine.table.remove(c)
        
        self.engine.last_capture_player = player
        
        # Aggiorna maschere
        self._captured_masks[player][card.index] = 1
        for c in taken:
            self._captured_masks[player][c.index] = 1
        
        # Scopa check
        if not self.engine.table and len(self.engine.deck) > 0:
            self.engine.scope[player] += 1
    
    def _do_opponent_turn(self) -> Optional[Tuple[Card, List[Card]]]:
        """Esegue il turno dell'avversario."""
        if not self.engine.hands[1]:
            return None
        
        mode = self.opponent_mode
        if mode == "mixed":
            mode = random.choice(["random", "self", "heuristic"])
        
        opp_moves = self.engine.get_legal_moves(1)
        
        if mode == "heuristic":
            opp_move = self._get_heuristic_move(opp_moves)
        elif mode == "self" and self.model is not None:
            opp_move = self._get_self_play_move(opp_moves)
        else:
            opp_move = random.choice(opp_moves)
        
        card, taken = opp_move
        self.engine.hands[1].remove(card)
        
        # Registra la carta nel history buffer (con safety check)
        if self.history_idx < HISTORY_BUFFER_SIZE:
            self.history[self.history_idx] = card.index / 39.0
            self.history_idx += 1
        
        if taken:
            self._execute_capture(1, card, taken)
        else:
            self.engine.table.append(card)
        
        self._opponent_played_this_hand[card.index] = 1
        
        return opp_move
    
    def _get_heuristic_move(self, moves: List[Tuple[Card, List[Card]]]) -> Tuple[Card, List[Card]]:
        """Sceglie mossa con euristica: Scopa > Settebello > Denari > Max carte."""
        def score(move: Tuple[Card, List[Card]]) -> int:
            card, taken = move
            if not taken:
                return -100
            
            s = len(taken) * 10
            
            # Scopa bonus
            if len(taken) == len(self.engine.table) and self.engine.deck:
                s += 500
            
            for c in taken:
                if c.rank == 7 and c.suit == Suit.DENARI:
                    s += 200
                elif c.suit == Suit.DENARI:
                    s += 20
                elif c.rank == 7:
                    s += 15
            
            return s
        
        return max(moves, key=score)
    
    def _get_self_play_move(self, moves: List[Tuple[Card, List[Card]]]) -> Tuple[Card, List[Card]]:
        """
        Usa il modello per scegliere la mossa dell'avversario.
        Implementa Fase 0 (selezione carta) e Fase 1 (selezione presa) come l'AI.
        """
        if self.model is None:
            return random.choice(moves)
        
        # === FASE 0: Scegli quale carta giocare ===
        obs = self._get_obs_for_opponent()
        
        mask = np.zeros(40, dtype=bool)
        for card in self.engine.hands[1]:
            mask[card.index] = True
        
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=False)
        
        card = next((c for c in self.engine.hands[1] if c.index == action), None)
        if card is None:
            return random.choice(moves)
        
        # Trova opzioni di presa per questa carta
        capture_options = self._get_valid_captures(card)
        
        if len(capture_options) == 0:
            # Nessuna presa: cala sul tavolo
            return (card, [])
        
        elif len(capture_options) == 1:
            # Una sola opzione: esegui automaticamente
            return (card, capture_options[0])
        
        else:
            # === FASE 1: Scegli quali carte prendere (iterativo con modello) ===
            selected: List[Card] = []
            target_rank = card.rank
            available_cards = list(self.engine.table)
            
            while True:
                current_sum = sum(c.rank for c in selected)
                
                if current_sum == target_rank:
                    # Somma raggiunta!
                    break
                
                needed = target_rank - current_sum
                
                # Maschera per le carte selezionabili (intelligente, evita vicoli ciechi)
                mask_phase1 = np.zeros(40, dtype=bool)
                selected_indices = {c.index for c in selected}
                
                # Trova solo le carte che possono portare a una presa valida
                valid_next = set()
                for combo in capture_options:
                    combo_indices = {c.index for c in combo}
                    if selected_indices.issubset(combo_indices):
                        remaining = combo_indices - selected_indices
                        for combo_card in combo:
                            if combo_card.index in remaining and combo_card.rank <= needed:
                                valid_next.add(combo_card.index)
                
                for idx in valid_next:
                    mask_phase1[idx] = True
                
                # Se nessuna carta valida, esci
                if not mask_phase1.any():
                    break
                
                # Crea observation aggiornata per Fase 1
                obs_phase1 = self._get_obs_for_opponent_phase1(card, selected)
                
                # Chiedi al modello
                action, _ = self.model.predict(obs_phase1, action_masks=mask_phase1, deterministic=False)
                
                # Aggiungi la carta selezionata
                selected_card = next(
                    (c for c in available_cards if c.index == action and c not in selected),
                    None
                )
                if selected_card:
                    selected.append(selected_card)
                else:
                    break  # Errore, esci
            
            # Verifica che la selezione sia valida
            if sum(c.rank for c in selected) == target_rank and selected:
                return (card, selected)
            else:
                # Fallback: usa la prima opzione valida
                return (card, capture_options[0])
    
    def _get_obs_for_opponent_phase1(self, card_played: Card, selected: List[Card]) -> np.ndarray:
        """Observation per l'avversario in Fase 1 (include history + starter)."""
        obs = self._get_obs_for_opponent()
        
        # Aggiorna per Fase 1
        # Indici: 253 = capture_mode, 254 = card_played_idx
        obs[253] = 1.0  # capture_mode = 1 (Fase 1)
        obs[254] = card_played.index / 39.0
        
        # Aggiorna selected_obs (posizione 200-239)
        for card in selected:
            obs[200 + card.index] = 1.0
        
        # History (255-294) e starter (295) già inclusi da _get_obs_for_opponent
        return obs
    
    def _get_obs_for_opponent(self) -> np.ndarray:
        """Crea observation swappata per l'avversario (per self-play)."""
        # Swap P0 <-> P1 nella osservazione
        obs = self._get_obs().copy()
        
        # Swap captured (indices 80-119 <-> 120-159)
        temp = obs[80:120].copy()
        obs[80:120] = obs[120:160]
        obs[120:160] = temp
        
        # Swap opponent_played (indices 160-199)
        # Dal punto di vista dell'avversario (P1), le carte "giocate dall'opponent"
        # sono quelle giocate dall'AI (P0). Usiamo _ai_played_this_hand per questo.
        obs[160:200] = self._ai_played_this_hand.astype(np.float32)
        
        # Swap stats (indices 241-245 <-> 246-250)
        temp = obs[241:246].copy()
        obs[241:246] = obs[246:251]
        obs[246:251] = temp
        
        # Swap hand size (index 251)
        # Originale: len(hands[1])/3.0 (mano avversario)
        # Per l'opponent: dovrebbe vedere len(hands[0])/3.0 (mano AI come "avversario")
        obs[251] = len(self.engine.hands[0]) / 3.0
        
        # Flip last_capture (index 252)
        obs[252] = 1.0 - obs[252]
        
        # History buffer (255-294) rimane invariato - cronologia condivisa
        
        # Flip starter_player (index 295) per prospettiva avversario
        obs[295] = 1.0 - obs[295]
        
        return obs
    
    def _check_deal_new_hand(self) -> None:
        """Distribuisce nuove carte se necessario."""
        if not self.engine.hands[0] and not self.engine.hands[1] and self.engine.deck:
            self.engine.deal_new_hand()
            self._opponent_played_this_hand = np.zeros(40, dtype=np.int8)
            self._ai_played_this_hand = np.zeros(40, dtype=np.int8)
    
    def _finish_step(
        self,
        ai_move: Tuple[Card, List[Card]],
        opp_move: Optional[Tuple[Card, List[Card]]],
        extra_reward: float = 0.0
    ) -> Tuple[ObsType, float, bool, bool, InfoDict]:
        """Conclude lo step e calcola reward."""
        terminated = False
        reward = extra_reward
        
        if self.engine.is_game_over():
            terminated = True
            self.engine.finalize_game()
            scores = self.engine.calculate_score()
            reward += float(scores[0]) - float(scores[1])
        
        info = {
            "ai_move": ai_move,
            "opponent_move": opp_move,
            "phase": self.phase,
        }
        
        return self._get_obs(), reward, terminated, False, info
    
    def _render_ansi(self) -> str:
        """Render testuale per debug."""
        lines = [
            f"Deck: {len(self.engine.deck)} | Table: {self.engine.table}",
            f"Hand P0: {self.engine.hands[0]}",
            f"Scope: {self.engine.scope} | Captured: {[len(c) for c in self.engine.captured]}",
        ]
        return "\n".join(lines)
