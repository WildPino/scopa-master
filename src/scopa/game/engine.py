"""
Scopa Engine - Motore di gioco con regole complete della Scopa.

Gestisce:
- Distribuzione carte
- Validazione mosse
- Calcolo prese (singole e combinazioni)
- Calcolo punteggio (Carte, Denari, Settebello, Primiera, Scope)
"""
from __future__ import annotations

import random
from itertools import combinations
from typing import List, Tuple, Optional

from scopa.game.cards import Card, Suit, create_deck
from scopa.config import PRIMIERA_VALUES


# Type alias per le mosse
Move = Tuple[Card, List[Card]]


class ScopaEngine:
    """
    Motore di gioco per la Scopa.
    
    Gestisce lo stato completo di una partita tra due giocatori.
    Player 0 = AI/Giocatore principale
    Player 1 = Avversario
    """
    
    def __init__(self):
        self.deck: List[Card] = []
        self.table: List[Card] = []
        self.hands: List[List[Card]] = [[], []]
        self.captured: List[List[Card]] = [[], []]
        self.scope: List[int] = [0, 0]
        self.last_capture_player: int = 0
        self.current_player: int = 0
    
    def reset(self) -> None:
        """Inizia una nuova smazzata completa (40 carte)."""
        self.deck = create_deck()
        random.shuffle(self.deck)
        
        # 4 carte sul tavolo
        self.table = [self.deck.pop() for _ in range(4)]
        
        # 3 carte per giocatore
        self.hands = [
            [self.deck.pop() for _ in range(3)],
            [self.deck.pop() for _ in range(3)]
        ]
        
        self.captured = [[], []]
        self.scope = [0, 0]
        self.current_player = 0
        self.last_capture_player = 0
    
    def get_legal_moves(self, player_index: int) -> List[Move]:
        """
        Ritorna tutte le mosse legali per un giocatore.
        
        Segue le regole della Scopa:
        - Se c'è una carta di rank uguale sul tavolo, DEVI prenderla (singola)
        - Altrimenti puoi prendere combinazioni che sommano al rank
        - Se non puoi prendere nulla, devi calare la carta
        
        Args:
            player_index: Indice del giocatore (0 o 1)
            
        Returns:
            Lista di mosse (carta_giocata, [carte_prese])
        """
        hand = self.hands[player_index]
        legal_moves: List[Move] = []
        
        for card in hand:
            # 1. Prese singole (obbligatorie se esistono)
            direct_matches = [t for t in self.table if t.rank == card.rank]
            
            if direct_matches:
                # Ogni match diretto è un'opzione separata
                for match in direct_matches:
                    legal_moves.append((card, [match]))
            else:
                # 2. Cerca combinazioni (somme)
                found_combo = False
                for r in range(2, len(self.table) + 1):
                    for combo in combinations(self.table, r):
                        if sum(c.rank for c in combo) == card.rank:
                            legal_moves.append((card, list(combo)))
                            found_combo = True
                
                # 3. Nessuna presa: cala sul tavolo
                if not found_combo:
                    legal_moves.append((card, []))
        
        return legal_moves
    
    def apply_move(self, player_index: int, card_played: Card, cards_taken: List[Card]) -> bool:
        """
        Esegue una mossa e aggiorna lo stato del gioco.
        
        Args:
            player_index: Indice del giocatore
            card_played: Carta giocata dalla mano
            cards_taken: Carte prese dal tavolo (vuoto = calata)
            
        Returns:
            True se è stata fatta una scopa
        """
        # Rimuovi carta dalla mano
        self.hands[player_index].remove(card_played)
        
        made_scopa = False
        
        if not cards_taken:
            # Calata sul tavolo
            self.table.append(card_played)
        else:
            # Presa
            self.captured[player_index].append(card_played)
            self.captured[player_index].extend(cards_taken)
            
            for c in cards_taken:
                self.table.remove(c)
            
            self.last_capture_player = player_index
            
            # Controllo Scopa (tavolo vuoto e mazzo non finito)
            if not self.table and len(self.deck) > 0:
                self.scope[player_index] += 1
                made_scopa = True
        
        return made_scopa
    
    def deal_new_hand(self) -> bool:
        """
        Distribuisce 3 nuove carte a testa se il mazzo non è vuoto.
        
        Returns:
            True se sono state distribuite carte, False se mazzo vuoto
        """
        if not self.deck:
            return False
        
        for i in range(2):
            self.hands[i] = [self.deck.pop() for _ in range(min(3, len(self.deck)))]
        
        return True
    
    def finalize_game(self) -> None:
        """
        Finalizza la partita: le carte rimaste sul tavolo
        vanno all'ultimo giocatore che ha preso.
        """
        self.captured[self.last_capture_player].extend(self.table)
        self.table = []
    
    def calculate_score(self) -> Tuple[int, int]:
        """
        Calcola i punteggi finali (4 punti di mazzo + scope).
        
        Punti assegnati per:
        - Carte: chi ha più di 20 carte
        - Denari: chi ha più di 5 denari
        - Settebello: chi ha il 7 di denari
        - Primiera: chi ha il punteggio primiera più alto
        - Scope: 1 punto per ogni scopa
        
        Returns:
            Tuple (score_player_0, score_player_1)
        """
        scores = list(self.scope)  # Partenza dalle scope
        
        # 1. Carte (Lungo)
        n_cards = [len(self.captured[i]) for i in range(2)]
        if n_cards[0] > 20:
            scores[0] += 1
        elif n_cards[1] > 20:
            scores[1] += 1
        
        # 2. Denari
        n_denari = [
            sum(1 for c in self.captured[i] if c.suit == Suit.DENARI)
            for i in range(2)
        ]
        if n_denari[0] > 5:
            scores[0] += 1
        elif n_denari[1] > 5:
            scores[1] += 1
        
        # 3. Settebello (7 di Denari)
        for i in range(2):
            if any(c.rank == 7 and c.suit == Suit.DENARI for c in self.captured[i]):
                scores[i] += 1
                break
        
        # 4. Primiera
        primiera_scores = [self._get_primiera_score(self.captured[i]) for i in range(2)]
        if primiera_scores[0] > primiera_scores[1]:
            scores[0] += 1
        elif primiera_scores[1] > primiera_scores[0]:
            scores[1] += 1
        
        return (scores[0], scores[1])
    
    def _get_primiera_score(self, captured_cards: List[Card]) -> int:
        """
        Calcola il punteggio primiera per un insieme di carte.
        
        La primiera usa i valori: 7=21, 6=18, 1=16, 5=15, 4=14, 3=13, 2=12, 8/9/10=10
        Per ogni seme conta solo la carta con valore più alto.
        
        Args:
            captured_cards: Carte catturate
            
        Returns:
            Somma dei migliori valori per ogni seme (max 84)
        """
        best_per_suit = [0, 0, 0, 0]
        
        for card in captured_cards:
            val = PRIMIERA_VALUES[card.rank]
            if val > best_per_suit[card.suit]:
                best_per_suit[card.suit] = val
        
        return sum(best_per_suit)
    
    def is_game_over(self) -> bool:
        """Controlla se la partita è terminata."""
        return (
            not self.hands[0] and 
            not self.hands[1] and 
            not self.deck
        )
    
    def get_state_summary(self) -> dict:
        """Ritorna un riepilogo dello stato per debug/logging."""
        return {
            "deck_remaining": len(self.deck),
            "table": len(self.table),
            "hands": [len(self.hands[i]) for i in range(2)],
            "captured": [len(self.captured[i]) for i in range(2)],
            "scope": list(self.scope),
            "last_capture": self.last_capture_player,
        }
