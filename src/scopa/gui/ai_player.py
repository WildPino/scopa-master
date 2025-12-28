"""
AI Player - Uses trained MaskablePPO model to play Scopa.

Falls back to heuristic AI if sb3_contrib is not available.
"""
from __future__ import annotations

import random
from typing import Optional, Tuple, List, Any

import numpy as np

from scopa.game import Card, Suit, ScopaEngine
from scopa.config import MODELS_DIR


# Try to import sb3_contrib
SB3_AVAILABLE = False
try:
    from sb3_contrib import MaskablePPO
    SB3_AVAILABLE = True
except ImportError:
    pass


class AIPlayer:
    """
    AI player that uses trained MaskablePPO model or heuristic fallback.
    
    Args:
        model_path: Path to trained model .zip file
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.use_model = False
        self.model: Any = None
        
        if SB3_AVAILABLE and model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str) -> None:
        """Load model from .zip file."""
        import os
        
        try:
            if model_path.endswith(".zip"):
                model_path = model_path[:-4]
            
            if os.path.exists(model_path + ".zip"):
                self.model = MaskablePPO.load(model_path)
                self.use_model = True
                print("✅ Modello AI caricato!")
            else:
                print(f"⚠️ Modello non trovato: {model_path}.zip")
        except Exception as e:
            print(f"⚠️ Errore caricamento: {e}")
            self.use_model = False
    
    def get_observation(self, engine: ScopaEngine, player_index: int = 1) -> np.ndarray:
        """Create observation vector for the AI."""
        obs = np.zeros(255, dtype=np.float32)
        opp_index = 1 - player_index
        
        # Hand (0-39)
        for card in engine.hands[player_index]:
            obs[card.index] = 1.0
        
        # Table (40-79)
        for card in engine.table:
            obs[40 + card.index] = 1.0
        
        # Own captured (80-119)
        for card in engine.captured[player_index]:
            obs[80 + card.index] = 1.0
        
        # Opponent captured (120-159)
        for card in engine.captured[opp_index]:
            obs[120 + card.index] = 1.0
        
        # Stats (240-254)
        obs[240] = len(engine.deck) / 40.0
        obs[241] = engine.scope[player_index] / 10.0
        obs[242] = len(engine.captured[player_index]) / 40.0
        obs[243] = sum(1 for c in engine.captured[player_index] if c.suit == Suit.DENARI) / 10.0
        obs[244] = 1.0 if any(c.rank == 7 and c.suit == Suit.DENARI for c in engine.captured[player_index]) else 0.0
        obs[245] = engine._get_primiera_score(engine.captured[player_index]) / 84.0
        obs[246] = engine.scope[opp_index] / 10.0
        obs[247] = len(engine.captured[opp_index]) / 40.0
        obs[248] = sum(1 for c in engine.captured[opp_index] if c.suit == Suit.DENARI) / 10.0
        obs[249] = 1.0 if any(c.rank == 7 and c.suit == Suit.DENARI for c in engine.captured[opp_index]) else 0.0
        obs[250] = engine._get_primiera_score(engine.captured[opp_index]) / 84.0
        obs[251] = len(engine.hands[opp_index]) / 3.0
        obs[252] = float(engine.last_capture_player)
        
        return obs
    
    def choose_card(self, engine: ScopaEngine, player_index: int = 1) -> Optional[Card]:
        """Choose which card to play."""
        if self.use_model and SB3_AVAILABLE:
            return self._choose_neural(engine, player_index)
        return self._choose_heuristic(engine, player_index)
    
    def _choose_neural(self, engine: ScopaEngine, player_index: int) -> Optional[Card]:
        """Choose using neural network."""
        obs = self.get_observation(engine, player_index)
        
        mask = np.zeros(40, dtype=bool)
        for card in engine.hands[player_index]:
            mask[card.index] = True
        
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        
        for card in engine.hands[player_index]:
            if card.index == int(action):
                return card
        
        return engine.hands[player_index][0] if engine.hands[player_index] else None
    
    def _choose_heuristic(self, engine: ScopaEngine, player_index: int) -> Optional[Card]:
        """Choose using heuristic strategy."""
        moves = engine.get_legal_moves(player_index)
        if not moves:
            return None
        
        def score(move: Tuple[Card, List[Card]]) -> int:
            card, capture = move
            if not capture:
                return -100
            
            s = len(capture) * 10
            
            # Scopa bonus
            if len(capture) == len(engine.table) and engine.deck:
                s += 500
            
            for c in capture:
                if c.rank == 7 and c.suit == Suit.DENARI:
                    s += 200
                elif c.suit == Suit.DENARI:
                    s += 20
                elif c.rank == 7:
                    s += 15
            
            return s
        
        return max(moves, key=score)[0]
    
    def choose_capture(
        self,
        engine: ScopaEngine,
        card_played: Card,
        valid_captures: List[List[Card]]
    ) -> List[Card]:
        """Choose which capture combination to use."""
        if not valid_captures:
            return []
        if len(valid_captures) == 1:
            return valid_captures[0]
        
        def score(capture: List[Card]) -> int:
            s = 0
            for card in capture:
                if card.rank == 7 and card.suit == Suit.DENARI:
                    s += 100
                elif card.suit == Suit.DENARI:
                    s += 10
                elif card.rank == 7:
                    s += 5
            return s + len(capture)
        
        return max(valid_captures, key=score)
    
    def get_move(
        self,
        engine: ScopaEngine,
        player_index: int = 1
    ) -> Tuple[Optional[Card], List[Card]]:
        """Get the full move (card + capture)."""
        card = self.choose_card(engine, player_index)
        if card is None:
            return None, []
        
        moves = engine.get_legal_moves(player_index)
        captures = [c for m, c in moves if m.index == card.index]
        
        if captures:
            capture = self.choose_capture(engine, card, captures)
        else:
            capture = []
        
        return card, capture
