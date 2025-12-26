"""
Visual Match - Visualizzazione partita Scopa con sistema a due fasi
"""
import numpy as np
from scopa_env import ScopaEnv
from scopa_engine import Suit

def format_card(card):
    suits = ["Co", "De", "Ba", "Sp"]  # Coppe, Denari, Bastoni, Spade
    return f"{card.rank}{suits[card.suit]}"

def format_cards(cards):
    if not cards:
        return "[]"
    return "[" + ", ".join(format_card(c) for c in cards) + "]"

def print_state(env, turn):
    print(f"\n{'='*50}")
    print(f"TURNO {turn} | Fase: {env.phase}")
    print(f"{'='*50}")
    print(f"Mazzo: {len(env.engine.deck)} | Tavolo: {format_cards(env.engine.table)}")
    print(f"Mano AI: {format_cards(env.engine.hands[0])} | Mano Avv: {len(env.engine.hands[1])} carte")
    print(f"Prese AI: {len(env.engine.captured[0])} | Prese Avv: {len(env.engine.captured[1])}")
    print(f"Scope AI: {env.engine.scope[0]} | Scope Avv: {env.engine.scope[1]}")
    
    if env.phase == 1:
        print(f">>> FASE 1: Carta giocata = {format_card(env.card_played)}")
        print(f">>> Selezionate: {format_cards(env.selected_captures)} (sum={sum(c.rank for c in env.selected_captures)})")

def play_match():
    env = ScopaEnv()
    obs, _ = env.reset()
    
    print("\n" + "="*50)
    print("       NUOVA PARTITA DI SCOPA (Due Fasi)")
    print("="*50)
    
    turn = 0
    done = False
    
    while not done:
        turn += 1
        print_state(env, turn)
        
        mask = env.action_masks()
        valid_actions = np.where(mask)[0]
        
        if len(valid_actions) == 0:
            print("ERRORE: Nessuna azione valida!")
            break
        
        action = np.random.choice(valid_actions)
        
        # Mostra l'azione scelta
        if env.phase == 0:
            card = next((c for c in env.engine.hands[0] if c.index == action), None)
            if card:
                print(f"\n-> AI gioca: {format_card(card)}")
        else:
            if action == 40:
                print(f"\n-> AI conferma presa")
            else:
                card = next((c for c in env.engine.table if c.index == action), None)
                if card:
                    print(f"\n-> AI seleziona: {format_card(card)}")
        
        obs, reward, done, _, info = env.step(action)
        
        # Mostra risultato
        if info.get('ai_move') and info['ai_move'][0]:
            ai_card, ai_taken = info['ai_move']
            if ai_taken:
                print(f"   Presa: {format_cards(ai_taken)}")
        
        if info.get('opponent_move'):
            opp_card, opp_taken = info['opponent_move']
            if opp_taken:
                print(f"   Avversario prende: {format_card(opp_card)} -> {format_cards(opp_taken)}")
            else:
                print(f"   Avversario cala: {format_card(opp_card)}")
        
        if reward != 0:
            print(f"   REWARD: {reward}")
        
        input("\n[INVIO per continuare...]")
    
    # Fine partita
    print("\n" + "="*50)
    print("PARTITA TERMINATA!")
    print("="*50)
    
    scores = env.engine.calculate_score()
    print(f"\nPUNTEGGIO: AI {scores[0]} - {scores[1]} Avversario")
    
    print(f"\nSTATISTICHE:")
    print(f"  Carte: AI {len(env.engine.captured[0])} | Avv {len(env.engine.captured[1])}")
    print(f"  Scope: AI {env.engine.scope[0]} | Avv {env.engine.scope[1]}")
    
    denari_ai = sum(1 for c in env.engine.captured[0] if c.suit == Suit.DENARI)
    denari_avv = sum(1 for c in env.engine.captured[1] if c.suit == Suit.DENARI)
    print(f"  Denari: AI {denari_ai} | Avv {denari_avv}")
    
    sette_ai = any(c.rank == 7 and c.suit == Suit.DENARI for c in env.engine.captured[0])
    print(f"  Settebello: {'AI' if sette_ai else 'Avversario'}")
    
    p_ai = env.engine._get_primiera_score(env.engine.captured[0])
    p_avv = env.engine._get_primiera_score(env.engine.captured[1])
    print(f"  Primiera: AI {p_ai} | Avv {p_avv}")
    
    if scores[0] > scores[1]:
        print("\nAI VINCE!")
    elif scores[1] > scores[0]:
        print("\nAVVERSARIO VINCE!")
    else:
        print("\nPAREGGIO!")

if __name__ == "__main__":
    play_match()