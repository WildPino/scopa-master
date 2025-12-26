import random
from itertools import combinations
from enum import IntEnum

class Suit(IntEnum):
    COPPE = 0
    DENARI = 1
    BASTONI = 2
    SPADE = 3

class Card:
    def __init__(self, rank, suit):
        self.rank = rank  # 1-10
        self.suit = suit  # 0-3 (Suit Enum)
        self.index = (suit * 10) + (rank - 1)  # 0-39 (Indice univoco per l'AI)

    def __repr__(self):
        suits = ["🏆", "🪙", "🪵", "🗡️"]
        ranks = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        return f"{ranks[self.rank-1]}{suits[self.suit]}"

    def __eq__(self, other):
        return self.index == other.index

class ScopaEngine:
    def __init__(self):
        self.deck = [Card(r, s) for s in range(4) for r in range(1, 11)]
        self.table = []
        self.hands = [[], []]
        self.captured = [[], []] # Carte prese da P1 e P2
        self.scope = [0, 0]      # Scope fatte da P1 e P2
        self.last_capture_player = 0 # Chi ha preso l'ultima volta
        self.current_player = 0

    def reset(self):
        """Inizia una smazzata completa (40 carte)"""
        self.deck = [Card(r, s) for s in range(4) for r in range(1, 11)]
        random.shuffle(self.deck)
        self.table = [self.deck.pop() for _ in range(4)]
        self.hands = [[self.deck.pop() for _ in range(3)], 
                      [self.deck.pop() for _ in range(3)]]
        self.captured = [[], []]
        self.scope = [0, 0]
        self.current_player = 0
        self.last_capture_player = 0

    def get_legal_moves(self, player_index):
        """
        Ritorna una lista di tuple: (carta_giocata, [lista_carte_prese])
        Segue rigorosamente le regole della Scopa.
        """
        hand = self.hands[player_index]
        legal_moves = []

        for card in hand:
            # 1. Trova prese singole (Regola: se c'è una carta uguale, DEVI prendere quella)
            direct_matches = [t_card for t_card in self.table if t_card.rank == card.rank]
            
            if direct_matches:
                # Obbligatorio prendere la singola (se ce n'è più di una di quel rango, scegli quale)
                for match in direct_matches:
                    legal_moves.append((card, [match]))
            else:
                # 2. Trova combinazioni (Somme)
                possible_combinations = []
                for r in range(2, len(self.table) + 1):
                    for combo in combinations(self.table, r):
                        if sum(c.rank for c in combo) == card.rank:
                            possible_combinations.append(list(combo))
                
                if possible_combinations:
                    for combo in possible_combinations:
                        legal_moves.append((card, combo))
                else:
                    # 3. Nessuna presa: la carta viene calata sul tavolo
                    legal_moves.append((card, []))
        
        return legal_moves

    def apply_move(self, player_index, card_played, cards_taken):
        """Esegue la mossa e aggiorna lo stato del gioco"""
        # Rimuovi carta dalla mano
        self.hands[player_index].remove(card_played)

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
            
            # Controllo Scopa
            if not self.table and len(self.deck) > 0:
                self.scope[player_index] += 1

    def deal_new_hand(self):
        """Distribuisce 3 nuove carte a testa se il mazzo non è vuoto"""
        if self.deck:
            for i in range(2):
                self.hands[i] = [self.deck.pop() for _ in range(3)]
            return True
        return False

    def finalize_game(self):
        """A fine mazzo, le carte rimaste in tavola vanno all'ultimo che ha preso"""
        self.captured[self.last_capture_player].extend(self.table)
        self.table = []

    def calculate_score(self):
        """Calcola i 4 punti di mazzo + scope"""
        scores = [self.scope[0], self.scope[1]]
        
        # 1. Carte (Lungo)
        if len(self.captured[0]) > 20: scores[0] += 1
        elif len(self.captured[1]) > 20: scores[1] += 1
        
        # 2. Denari
        d0 = sum(1 for c in self.captured[0] if c.suit == Suit.DENARI)
        d1 = sum(1 for c in self.captured[1] if c.suit == Suit.DENARI)
        if d0 > 5: scores[0] += 1
        elif d1 > 5: scores[1] += 1
        
        # 3. Settebello
        if any(c.rank == 7 and c.suit == Suit.DENARI for c in self.captured[0]):
            scores[0] += 1
        else:
            scores[1] += 1
            
        # 4. Primiera
        p0 = self._get_primiera_score(self.captured[0])
        p1 = self._get_primiera_score(self.captured[1])
        if p0 > p1: scores[0] += 1
        elif p1 > p0: scores[1] += 1
        
        return scores

    def _get_primiera_score(self, captured_cards):
        # Valori Primiera: 7=21, 6=18, 1=16, 5=15, 4=14, 3=13, 2=12, 8/9/10=10
        values = {7:21, 6:18, 1:16, 5:15, 4:14, 3:13, 2:12, 8:10, 9:10, 10:10}
        best_per_suit = [0, 0, 0, 0]
        for c in captured_cards:
            val = values[c.rank]
            if val > best_per_suit[c.suit]:
                best_per_suit[c.suit] = val
        return sum(best_per_suit)