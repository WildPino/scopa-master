"""
AI Player - Uses trained MaskablePPO model from Stable Baselines 3 to play Scopa
Falls back to heuristic AI if sb3_contrib is not installed.
"""
import numpy as np
import os
import sys
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scopa_engine import ScopaEngine, Suit

# Try to import sb3_contrib for the trained model
SB3_AVAILABLE = False
try:
    from sb3_contrib import MaskablePPO
    SB3_AVAILABLE = True
    print("✅ sb3_contrib disponibile - usando modello addestrato")
except ImportError:
    print("⚠️ sb3_contrib non disponibile - L'AI userà una strategia euristica")


class AIPlayer:
    """AI player that uses the trained MaskablePPO model or heuristic fallback"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the AI player.
        
        Args:
            model_path: Path to the trained model .zip file (without .zip extension)
        """
        self.use_model = False
        self.model = None
        
        if SB3_AVAILABLE and model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load model from .zip file using MaskablePPO"""
        try:
            # Remove .zip extension if present
            if model_path.endswith('.zip'):
                model_path = model_path[:-4]
            
            if os.path.exists(model_path + '.zip'):
                self.model = MaskablePPO.load(model_path)
                self.use_model = True
                print("✅ Modello AI caricato con successo!")
            else:
                print(f"⚠️ Modello non trovato: {model_path}.zip")
        except Exception as e:
            print(f"⚠️ Errore caricamento modello: {e}")
            self.use_model = False
    
    def get_observation(self, engine: ScopaEngine, player_index: int = 1) -> np.ndarray:
        """
        Create observation vector for the AI (from its perspective).
        Must match the observation space used during training (255 elements).
        
        Structure (same as scopa_env.py):
        - 0-39: Hand (40 bits)
        - 40-79: Table (40 bits)
        - 80-119: Own captured cards (40 bits)
        - 120-159: Opponent captured cards (40 bits)
        - 160-199: Opponent played cards this hand (40 bits) - not tracked here
        - 200-239: Selected captures (40 bits) - Phase 1 only
        - 240: Deck remaining (normalized)
        - 241-251: Stats (11 values)
        - 252: Last capture player
        - 253: Capture mode (phase)
        - 254: Card played index
        """
        obs = np.zeros(255, dtype=np.float32)
        
        # AI's hand (0-39)
        for card in engine.hands[player_index]:
            obs[card.index] = 1
        
        # Table cards (40-79)
        for card in engine.table:
            obs[40 + card.index] = 1
        
        # Own captured cards (80-119)
        for card in engine.captured[player_index]:
            obs[80 + card.index] = 1
        
        # Opponent captured cards (120-159)
        opp_index = 1 - player_index
        for card in engine.captured[opp_index]:
            obs[120 + card.index] = 1
        
        # Opponent played cards - not tracked in GUI, leave as zeros (160-199)
        
        # Selected captures - Phase 0, so zeros (200-239)
        
        # Deck remaining (240)
        obs[240] = len(engine.deck) / 40.0
        
        # Stats (241-251) - 11 values
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
        
        # Last capture player (252)
        obs[252] = float(engine.last_capture_player)
        
        # Capture mode (253) - Phase 0 sempre per la GUI
        obs[253] = 0.0
        
        # Card played index (254) - Phase 0, non c'e' carta giocata
        obs[254] = 0.0
        
        return obs
    
    def get_action_mask(self, engine: ScopaEngine, player_index: int = 1) -> np.ndarray:
        """
        Get mask of valid card selections (40 actions).
        """
        mask = np.zeros(40, dtype=bool)
        
        # Cards in hand are valid to play
        for card in engine.hands[player_index]:
            mask[card.index] = True
        
        return mask
    
    def choose_card_neural(self, engine: ScopaEngine, player_index: int = 1):
        """Choose card using neural network model"""
        obs = self.get_observation(engine, player_index)
        mask = self.get_action_mask(engine, player_index)
        
        # Use MaskablePPO's predict method
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        action = int(action)
        
        # Find the card in hand with this index
        for card in engine.hands[player_index]:
            if card.index == action:
                return card
        
        # Fallback: return first card in hand
        return engine.hands[player_index][0] if engine.hands[player_index] else None
    
    def choose_card_heuristic(self, engine: ScopaEngine, player_index: int = 1):
        """
        Choose card using heuristic strategy.
        Priority: Scopa > Settebello > Denari > Max cards > Random
        """
        legal_moves = engine.get_legal_moves(player_index)
        
        if not legal_moves:
            return None
        
        def score_move(move):
            card_played, capture = move
            score = 0
            
            if not capture:
                # Dropping is less desirable
                return -100
            
            # Scopa is top priority
            would_clear_table = len(capture) == len(engine.table) and len(engine.deck) > 0
            if would_clear_table:
                score += 500
            
            for c in capture:
                # Settebello
                if c.rank == 7 and c.suit == Suit.DENARI:
                    score += 200
                # Other Denari
                elif c.suit == Suit.DENARI:
                    score += 20
                # 7s for Primiera
                elif c.rank == 7:
                    score += 15
                # 6s for Primiera
                elif c.rank == 6:
                    score += 10
            
            # More cards = better
            score += len(capture) * 5
            
            return score
        
        # Find best move
        best_move = max(legal_moves, key=score_move)
        return best_move[0]  # Return the card, not the capture
    
    def choose_card(self, engine: ScopaEngine, player_index: int = 1):
        """
        Choose which card to play.
        Uses neural network if available, otherwise heuristic.
        """
        if self.use_model and SB3_AVAILABLE:
            return self.choose_card_neural(engine, player_index)
        else:
            return self.choose_card_heuristic(engine, player_index)
    
    def choose_capture(self, engine: ScopaEngine, card_played, valid_captures: list):
        """
        Choose which capture combination to use.
        Uses heuristic: prioritize Settebello, then Denari, then most cards.
        """
        if not valid_captures:
            return []
        
        if len(valid_captures) == 1:
            return valid_captures[0]
        
        def score_capture(capture):
            score = 0
            for card in capture:
                # Settebello (7 di Denari) is most valuable
                if card.rank == 7 and card.suit == Suit.DENARI:
                    score += 100
                # Denari are valuable
                elif card.suit == Suit.DENARI:
                    score += 10
                # 7s are good for Primiera
                elif card.rank == 7:
                    score += 5
            # More cards is generally better
            score += len(capture)
            return score
        
        return max(valid_captures, key=score_capture)
    
    def get_move(self, engine: ScopaEngine, player_index: int = 1):
        """
        Get the full move (card to play + capture combination).
        
        Returns:
            Tuple of (card_played, cards_to_capture)
        """
        # Choose card to play
        card = self.choose_card(engine, player_index)
        if card is None:
            return None, []
        
        # Get valid captures for this card
        legal_moves = engine.get_legal_moves(player_index)
        valid_captures = [capture for c, capture in legal_moves if c.index == card.index]
        
        if valid_captures:
            capture = self.choose_capture(engine, card, valid_captures)
        else:
            capture = []
        
        return card, capture
