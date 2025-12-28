"""
Scopa Game Module - Core game logic.

Exports:
- Card: Rappresentazione di una carta napoletana
- Suit: Enum dei quattro semi
- ScopaEngine: Motore di gioco con regole complete
"""
from scopa.game.cards import Card, Suit
from scopa.game.engine import ScopaEngine

__all__ = ["Card", "Suit", "ScopaEngine"]
