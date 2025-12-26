"""
Scopa Environment - Sistema a Due Fasi

Fase 0: Selezione carta da giocare (40 azioni, mask = mano)
Fase 1: Selezione carte da prendere (iterativa, solo se multiple opzioni)

Observation Space: 255 elementi
Action Space: 41 azioni (0-39 = carta, 40 = conferma presa)
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import logging
from itertools import combinations
from scopa_engine import ScopaEngine, Suit

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


class ScopaEnv(gym.Env):
    """
    Ambiente Scopa con supporto per diverse modalità di avversario.
    
    opponent_mode:
        - 'random': Avversario gioca mosse casuali (default)
        - 'self': Self-play, usa lo stesso modello per l'avversario
        - 'heuristic': Avversario usa euristica (da implementare)
        - 'mixed': Alterna casualmente tra le modalità
    """
    
    def __init__(self, opponent_mode='random', model=None):
        super(ScopaEnv, self).__init__()
        self.engine = ScopaEngine()
        
        # Modalità avversario
        self.opponent_mode = opponent_mode
        self.model = model  # Usato per self-play
        self.available_modes = ['random', 'self', 'heuristic']

        # --- ACTION SPACE: 41 azioni ---
        # 0-39: Indice carta (giocare in Fase 0, prendere in Fase 1)
        # 40: Conferma presa (solo Fase 1)
        self.action_space = spaces.Discrete(41)

        # --- OBSERVATION SPACE: 255 elementi ---
        # - 40 bit: Mano Player 0 (AI)
        # - 40 bit: Tavolo 
        # - 40 bit: Carte prese da Player 0 (partita intera)
        # - 40 bit: Carte prese da Player 1 (partita intera)
        # - 40 bit: Carte giocate dall'avversario in questa mano
        # - 40 bit: Carte selezionate per la presa corrente (Fase 1)
        # - 1: Carte rimanenti nel mazzo (0.0 - 1.0)
        # - 11: Statistiche normalizzate
        # - 1: Last capture player (0/1)
        # - 1: Capture mode (0 = Fase 0, 1 = Fase 1)
        # - 1: Card played index (0-39, normalizzato)
        self.observation_space = spaces.Box(low=0, high=1, shape=(255,), dtype=np.float32)
        
        # Stato interno per le fasi
        self.phase = 0  # 0 = selezione carta, 1 = selezione presa
        self.card_played = None  # Carta giocata in Fase 0
        self.selected_captures = []  # Carte selezionate in Fase 1
        self.valid_capture_options = []  # Combinazioni valide per la presa
        
        # Maschere per le carte prese
        self.captured_masks = [np.zeros(40, dtype=np.int8), np.zeros(40, dtype=np.int8)]
        self.opponent_played_this_hand = np.zeros(40, dtype=np.int8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset()
        
        # Reset stato fasi
        self.phase = 0
        self.card_played = None
        self.selected_captures = []
        self.valid_capture_options = []
        
        # Reset maschere
        self.captured_masks = [np.zeros(40, dtype=np.int8), np.zeros(40, dtype=np.int8)]
        self.opponent_played_this_hand = np.zeros(40, dtype=np.int8)
        
        # Randomizza chi inizia (50% AI, 50% Avversario)
        # Se esce 1, l'avversario fa la prima mossa
        if random.random() < 0.5:
             # Inizia l'avversario
             opp_move = self._do_opponent_turn()
             # Non serve tracciare opp_move nel reset info, l'observation sarà aggiornata
             
        return self._get_obs(), {}

    def _get_obs(self):
        # 1. Mano AI (40)
        hand_obs = np.zeros(40)
        for card in self.engine.hands[0]:
            hand_obs[card.index] = 1
            
        # 2. Tavolo (40) - esclude le carte già selezionate in Fase 1
        table_obs = np.zeros(40)
        for card in self.engine.table:
            if card not in self.selected_captures:
                table_obs[card.index] = 1
            
        # 3. Carte prese P0 (40)
        captured_p0 = self.captured_masks[0].astype(np.float32)
        
        # 4. Carte prese P1 (40)
        captured_p1 = self.captured_masks[1].astype(np.float32)
        
        # 5. Carte giocate dall'avversario questa mano (40)
        opponent_played = self.opponent_played_this_hand.astype(np.float32)
        
        # 6. Mazzo (1)
        deck_obs = [len(self.engine.deck) / 40.0]

        # 7. Statistiche (11)
        stats = [
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
            len(self.engine.hands[1]) / 3.0 
        ]
        
        # 8. Last capture player (1)
        last_capture = [float(self.engine.last_capture_player)]
        
        # 9. Capture mode (1) - indica se siamo in Fase 1
        capture_mode = [1.0 if self.phase == 1 else 0.0]
        
        # 10. Card played index (1) - normalizzato 0-39 → 0-1
        card_played_idx = [self.card_played.index / 39.0 if self.card_played else 0.0]
        
        # 11. Carte selezionate per la presa (40 bit) - memoria esplicita per Fase 1
        selected_obs = np.zeros(40)
        for card in self.selected_captures:
            selected_obs[card.index] = 1
        
        return np.concatenate([
            hand_obs,           # 40
            table_obs,          # 40
            captured_p0,        # 40
            captured_p1,        # 40
            opponent_played,    # 40
            selected_obs,       # 40 - NUOVO: carte selezionate in Fase 1
            deck_obs,           # 1
            stats,              # 11
            last_capture,       # 1
            capture_mode,       # 1
            card_played_idx,    # 1
        ]).astype(np.float32)   # Totale: 255

    def _get_valid_captures(self, card):
        """
        Trova tutte le combinazioni valide di presa per una carta.
        Regola Scopa: se c'è una carta di rank uguale, DEVI prenderla (no somme).
        """
        # Presa diretta obbligatoria
        direct_matches = [c for c in self.engine.table if c.rank == card.rank]
        if direct_matches:
            # Ritorna ogni carta singola come opzione separata
            return [[c] for c in direct_matches]
        
        # Cerca combinazioni che sommano al rank
        valid_combos = []
        for r in range(2, len(self.engine.table) + 1):
            for combo in combinations(self.engine.table, r):
                if sum(c.rank for c in combo) == card.rank:
                    valid_combos.append(list(combo))
        
        return valid_combos

    def action_masks(self):
        """Ritorna la maschera delle azioni legali."""
        mask = np.zeros(41, dtype=bool)
        
        if self.phase == 0:
            # Fase 0: Mask per carte in mano
            for card in self.engine.hands[0]:
                mask[card.index] = True
        else:
            # Fase 1: Mask per carte sul tavolo che possono contribuire
            current_sum = sum(c.rank for c in self.selected_captures)
            needed = self.card_played.rank - current_sum
            
            for card in self.engine.table:
                if card not in self.selected_captures:
                    if card.rank <= needed:
                        mask[card.index] = True
            
            # Azione "conferma" (40) disponibile se:
            # 1. La somma è corretta (presa valida)
            # 2. OPPURE non ci sono altre azioni valide (escape hatch - terminerà con penalità)
            if current_sum == self.card_played.rank or not mask.any():
                mask[40] = True
                
        return mask

    def step(self, action):
        if self.phase == 0:
            return self._step_phase0(action)
        else:
            return self._step_phase1(action)

    def _step_phase0(self, action):
        """Fase 0: Selezione carta da giocare."""
        # Trova la carta selezionata
        card_to_play = next((c for c in self.engine.hands[0] if c.index == action), None)
        
        if card_to_play is None:
            logging.warning(f"FASE 0 - MOSSA INVALIDA! Azione {action} non corrisponde a carta in mano.")
            return self._get_obs(), -10.0, True, False, {"invalid_action": True, "phase": 0}
        
        self.card_played = card_to_play
        
        # Trova le opzioni di presa
        self.valid_capture_options = self._get_valid_captures(card_to_play)
        
        if len(self.valid_capture_options) == 0:
            # Nessuna presa possibile: cala sul tavolo
            self.engine.hands[0].remove(card_to_play)
            self.engine.table.append(card_to_play)
            
            ai_move = (card_to_play, [])
            opp_move = self._do_opponent_turn()
            self._check_deal_new_hand()
            
            return self._finish_step(ai_move, opp_move)
            
        elif len(self.valid_capture_options) == 1:
            # Una sola opzione: esegui automaticamente
            capture = self.valid_capture_options[0]
            # IMPORTANTE: Rimuovi la carta dalla mano PRIMA di aggiungerla alle prese
            self.engine.hands[0].remove(card_to_play)
            self._execute_capture(0, card_to_play, capture)
            
            ai_move = (card_to_play, capture)
            opp_move = self._do_opponent_turn()
            self._check_deal_new_hand()
            
            return self._finish_step(ai_move, opp_move)
            
        else:
            # Multiple opzioni: vai a Fase 1
            self.phase = 1
            self.selected_captures = []
            # Rimuovi la carta dalla mano (l'ha già "giocata")
            self.engine.hands[0].remove(card_to_play)
            
            # Reward intermedio neutro, episodio continua
            return self._get_obs(), 0.0, False, False, {"phase": 1, "awaiting_capture": True}

    def _step_phase1(self, action):
        """Fase 1: Selezione carte da prendere."""
        if action == 40:
            # Azione "conferma"
            current_sum = sum(c.rank for c in self.selected_captures)
            
            if current_sum == self.card_played.rank:
                # Presa valida!
                self._execute_capture(0, self.card_played, self.selected_captures)
                ai_move = (self.card_played, list(self.selected_captures))
                reward = 0.0
            else:
                # Somma errata: TERMINA EPISODIO (policy deve essere precisa)
                logging.warning(f"FASE 1 - Conferma con somma errata: {current_sum} != {self.card_played.rank}")
                return self._get_obs(), -10.0, True, False, {"invalid_action": True, "phase": 1, "reason": "wrong_sum"}
            
            self.phase = 0
            self.card_played = None
            self.selected_captures = []
            
            opp_move = self._do_opponent_turn()
            self._check_deal_new_hand()
            
            return self._finish_step(ai_move, opp_move, extra_reward=reward)
            
        else:
            # Selezione carta dal tavolo
            table_card = next((c for c in self.engine.table if c.index == action and c not in self.selected_captures), None)
            
            if table_card is None:
                # Mossa invalida: TERMINA EPISODIO
                logging.warning(f"FASE 1 - MOSSA INVALIDA! Carta {action} non sul tavolo.")
                return self._get_obs(), -10.0, True, False, {"invalid_action": True, "phase": 1, "reason": "card_not_on_table"}
            
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
                # Somma superata: TERMINA EPISODIO (non dovrebbe succedere con mask)
                logging.warning(f"FASE 1 - Somma superata: {current_sum} > {self.card_played.rank}")
                return self._get_obs(), -10.0, True, False, {"invalid_action": True, "phase": 1, "reason": "sum_exceeded"}
            
            else:
                # Somma non ancora raggiunta: continua Fase 1
                return self._get_obs(), 0.0, False, False, {"phase": 1, "selected_sum": current_sum}

    def _execute_capture(self, player_index, card_played, cards_taken):
        """
        Esegue una presa e aggiorna lo stato.
        Generalizzato per supportare sia AI (0) che avversario (1).
        
        NOTA: La carta deve essere già stata rimossa dalla mano del giocatore.
        """
        self.engine.captured[player_index].append(card_played)
        self.engine.captured[player_index].extend(cards_taken)
        
        for c in cards_taken:
            self.engine.table.remove(c)
        
        self.engine.last_capture_player = player_index
        
        # Aggiorna maschera prese
        self.captured_masks[player_index][card_played.index] = 1
        for c in cards_taken:
            self.captured_masks[player_index][c.index] = 1
        
        # Controllo Scopa (solo se tavolo vuoto e mazzo non finito)
        if not self.engine.table and len(self.engine.deck) > 0:
            self.engine.scope[player_index] += 1

    def _execute_random_valid_capture(self):
        """Fallback: esegue una presa random valida per l'AI."""
        if self.valid_capture_options:
            capture = random.choice(self.valid_capture_options)
            self._execute_capture(0, self.card_played, capture)

    def set_model(self, model):
        """Imposta il modello per self-play."""
        self.model = model
    
    def _do_opponent_turn(self):
        """Esegue il turno dell'avversario in base alla modalità."""
        if not self.engine.hands[1]:
            return None
        
        # Determina la modalità da usare
        mode = self.opponent_mode
        if mode == 'mixed':
            mode = random.choice(self.available_modes)
        
        # Ottieni le mosse legali
        opp_moves = self.engine.get_legal_moves(1)
        
        # Scegli la mossa in base alla modalità
        if mode == 'random':
            opp_move = random.choice(opp_moves)
            
        elif mode == 'self' and self.model is not None:
            # Self-play: usa il modello per scegliere
            # Crea observation dal punto di vista dell'avversario (swapped)
            opp_move = self._get_self_play_move(opp_moves)
            
        elif mode == 'heuristic':
            # Euristica: da implementare, per ora usa random
            opp_move = self._get_heuristic_move(opp_moves)
            
        else:
            # Fallback a random
            opp_move = random.choice(opp_moves)
        
        card_played, cards_taken = opp_move
        
        # Rimuovi carta dalla mano
        self.engine.hands[1].remove(card_played)
        
        if cards_taken:
            self._execute_capture(1, card_played, cards_taken)
        else:
            self.engine.table.append(card_played)
        
        # Traccia la carta giocata
        self.opponent_played_this_hand[card_played.index] = 1
        
        return opp_move
    
    def _get_self_play_move(self, opp_moves):
        """
        Usa il modello per scegliere la mossa dell'avversario.
        Implementa Fase 0 (selezione carta) e Fase 1 (selezione presa) come l'AI.
        """
        if self.model is None:
            return random.choice(opp_moves)
        
        # === FASE 0: Scegli quale carta giocare ===
        # Crea observation dal punto di vista dell'avversario (P1)
        obs_opp = self._get_obs_for_opponent()
        
        # Maschera per le carte in mano dell'avversario
        mask_phase0 = np.zeros(41, dtype=bool)
        for card in self.engine.hands[1]:
            mask_phase0[card.index] = True
        
        # Chiedi al modello quale carta giocare
        action, _ = self.model.predict(obs_opp, action_masks=mask_phase0, deterministic=False)
        
        # Trova la carta scelta
        card_to_play = next((c for c in self.engine.hands[1] if c.index == action), None)
        if card_to_play is None:
            return random.choice(opp_moves)
        
        # Trova le opzioni di presa per questa carta
        capture_options = self._get_valid_captures_for_card(card_to_play)
        
        if len(capture_options) == 0:
            # Nessuna presa: cala sul tavolo
            return (card_to_play, [])
        
        elif len(capture_options) == 1:
            # Una sola opzione: esegui automaticamente
            return (card_to_play, capture_options[0])
        
        else:
            # === FASE 1: Scegli quali carte prendere ===
            # Simula la selezione iterativa delle carte
            selected = []
            target_rank = card_to_play.rank
            available_cards = list(self.engine.table)
            
            while True:
                current_sum = sum(c.rank for c in selected)
                
                if current_sum == target_rank:
                    # Somma raggiunta!
                    break
                
                needed = target_rank - current_sum
                
                # Maschera per le carte selezionabili
                mask_phase1 = np.zeros(41, dtype=bool)
                for card in available_cards:
                    if card not in selected and card.rank <= needed:
                        mask_phase1[card.index] = True
                
                # Aggiungi "conferma" se somma corretta
                if current_sum == target_rank:
                    mask_phase1[40] = True
                
                # Se nessuna carta valida, esci (non dovrebbe succedere)
                if not mask_phase1.any():
                    break
                
                # Crea observation aggiornata
                obs_phase1 = self._get_obs_for_opponent_phase1(card_to_play, selected)
                
                # Chiedi al modello
                action, _ = self.model.predict(obs_phase1, action_masks=mask_phase1, deterministic=False)
                
                if action == 40:
                    # Conferma
                    break
                
                # Aggiungi la carta selezionata
                selected_card = next((c for c in available_cards if c.index == action and c not in selected), None)
                if selected_card:
                    selected.append(selected_card)
                else:
                    break  # Errore, esci
            
            # Verifica che la selezione sia valida
            if sum(c.rank for c in selected) == target_rank and selected:
                return (card_to_play, selected)
            else:
                # Fallback: usa la prima opzione valida
                return (card_to_play, capture_options[0])
    
    def _get_valid_captures_for_card(self, card):
        """Trova le combinazioni di presa valide per una carta (usato per opponent)."""
        # Presa diretta obbligatoria
        direct = [c for c in self.engine.table if c.rank == card.rank]
        if direct:
            return [[c] for c in direct]
        
        # Cerca somme
        from itertools import combinations
        valid = []
        for r in range(2, len(self.engine.table) + 1):
            for combo in combinations(self.engine.table, r):
                if sum(c.rank for c in combo) == card.rank:
                    valid.append(list(combo))
        return valid
    
    def _get_obs_for_opponent(self):
        """
        Crea observation dal punto di vista dell'avversario (P1).
        Scambia le informazioni di P0 e P1.
        """
        # Mano P1 (ora è "la mia mano")
        hand_obs = np.zeros(40)
        for card in self.engine.hands[1]:
            hand_obs[card.index] = 1
        
        # Tavolo (invariato)
        table_obs = np.zeros(40)
        for card in self.engine.table:
            table_obs[card.index] = 1
        
        # Prese scambiate: P1 diventa "le mie", P0 diventa "avversario"
        captured_mine = self.captured_masks[1].astype(np.float32)
        captured_opp = self.captured_masks[0].astype(np.float32)
        
        # Carte giocate dall'avversario (P0 dal punto di vista di P1)
        # Per semplicità, usiamo zero (P1 non traccia cosa ha tirato P0 in questa mano)
        opponent_played = np.zeros(40, dtype=np.float32)
        
        # Selected captures (fase 0, quindi vuoto)
        selected_obs = np.zeros(40)
        
        # Statistiche scambiate
        stats = [
            self.engine.scope[1] / 10.0,  # Le MIE scope (P1)
            len(self.engine.captured[1]) / 40.0,
            sum(1 for c in self.engine.captured[1] if c.suit == Suit.DENARI) / 10.0,
            1.0 if any(c.rank == 7 and c.suit == Suit.DENARI for c in self.engine.captured[1]) else 0.0,
            self.engine._get_primiera_score(self.engine.captured[1]) / 84.0,
            self.engine.scope[0] / 10.0,  # Scope avversario (P0)
            len(self.engine.captured[0]) / 40.0,
            sum(1 for c in self.engine.captured[0] if c.suit == Suit.DENARI) / 10.0,
            1.0 if any(c.rank == 7 and c.suit == Suit.DENARI for c in self.engine.captured[0]) else 0.0,
            self.engine._get_primiera_score(self.engine.captured[0]) / 84.0,
            len(self.engine.hands[0]) / 3.0  # Carte in mano avversario (P0)
        ]
        
        deck_obs = [len(self.engine.deck) / 40.0]
        last_capture = [1.0 - float(self.engine.last_capture_player)]  # Invertito
        capture_mode = [0.0]  # Fase 0
        card_played_idx = [0.0]
        
        return np.concatenate([
            hand_obs, table_obs, captured_mine, captured_opp, opponent_played,
            selected_obs, deck_obs, stats, last_capture, capture_mode, card_played_idx
        ]).astype(np.float32)
    
    def _get_obs_for_opponent_phase1(self, card_played, selected):
        """Observation per P1 in Fase 1."""
        obs = self._get_obs_for_opponent()
        
        # Aggiorna per Fase 1
        obs[240 + 12] = 1.0  # capture_mode = 1
        obs[240 + 13] = card_played.index / 39.0  # card_played_idx
        
        # Aggiorna selected_obs (posizione 200-239)
        for card in selected:
            obs[200 + card.index] = 1.0  # Assumendo che selected_obs sia a 200
        
        return obs
    
    def _get_heuristic_move(self, opp_moves):
        """
        Euristica semplice per l'avversario.
        Priorità: Settebello > Denari > Scope > Max carte
        """
        # TODO: implementare euristica avanzata
        # Per ora: prendi più carte possibile, preferisci denari
        def score_move(move):
            card, taken = move
            score = len(taken) * 10
            # Bonus per denari
            score += sum(1 for c in taken if c.suit == Suit.DENARI) * 5
            # Bonus per settebello
            if any(c.rank == 7 and c.suit == Suit.DENARI for c in taken):
                score += 50
            return score
        
        return max(opp_moves, key=score_move)

    def _check_deal_new_hand(self):
        """Distribuisce nuove carte se necessario."""
        if not self.engine.hands[0] and not self.engine.hands[1] and self.engine.deck:
            self.engine.deal_new_hand()
            self.opponent_played_this_hand = np.zeros(40, dtype=np.int8)

    def _finish_step(self, ai_move, opp_move, extra_reward=0.0):
        """Conclude lo step e calcola reward finale."""
        terminated = False
        reward = extra_reward
        
        if not self.engine.hands[0] and not self.engine.hands[1] and not self.engine.deck:
            terminated = True
            self.engine.finalize_game()
            scores = self.engine.calculate_score()
            reward += float(scores[0]) - float(scores[1])
        
        info = {
            "ai_move": ai_move,
            "opponent_move": opp_move,
            "phase": self.phase
        }
        
        return self._get_obs(), reward, terminated, False, info