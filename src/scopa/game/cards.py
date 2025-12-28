"""
Cards - Definizione delle classi Card e Suit.

Le carte napoletane sono composte da 4 semi (Coppe, Denari, Bastoni, Spade)
e 10 ranghi (1-10, dove 8=Fante, 9=Cavallo, 10=Re).
"""
from __future__ import annotations
from enum import IntEnum
from typing import List


class Suit(IntEnum):
    """Enum dei quattro semi delle carte napoletane."""
    COPPE = 0
    DENARI = 1
    BASTONI = 2
    SPADE = 3
    
    def __str__(self) -> str:
        symbols = ["🏆", "🪙", "🪵", "🗡️"]
        return symbols[self.value]


class Card:
    """
    Rappresenta una singola carta napoletana.
    
    Attributes:
        rank: Valore della carta (1-10)
        suit: Seme della carta (Suit enum)
        index: Indice univoco 0-39 per l'AI
    """
    
    __slots__ = ("rank", "suit", "index")
    
    def __init__(self, rank: int, suit: int | Suit):
        """
        Crea una nuova carta.
        
        Args:
            rank: Valore 1-10
            suit: Seme 0-3 o Suit enum
        """
        self.rank: int = rank
        self.suit: Suit = Suit(suit) if isinstance(suit, int) else suit
        self.index: int = (self.suit.value * 10) + (rank - 1)
    
    def __repr__(self) -> str:
        """Rappresentazione testuale con emoji."""
        ranks = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        return f"{ranks[self.rank - 1]}{self.suit}"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.index == other.index
    
    def __hash__(self) -> int:
        return hash(self.index)
    
    @classmethod
    def from_index(cls, index: int) -> Card:
        """
        Crea una carta dall'indice univoco.
        
        Args:
            index: Indice 0-39
            
        Returns:
            Card corrispondente
        """
        suit = index // 10
        rank = (index % 10) + 1
        return cls(rank, suit)


def create_deck() -> List[Card]:
    """
    Crea un mazzo completo di 40 carte napoletane.
    
    Returns:
        Lista di 40 Card ordinate per seme e rango
    """
    return [Card(r, s) for s in range(4) for r in range(1, 11)]
