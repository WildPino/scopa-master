"""
Visual Match - Visualizzazione partita Scopa in console.

Mostra una partita step-by-step con il sistema a due fasi.

Usage:
    python -m scopa.cli.visual_match
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from scopa.game import Card, Suit, ScopaEngine
from scopa.rl import ScopaEnv
from scopa.config import MODELS_DIR


def format_card(card: Card) -> str:
    """Formatta una carta per output testuale."""
    suits = ["Co", "De", "Ba", "Sp"]
    return f"{card.rank}{suits[card.suit]}"


def format_cards(cards: list) -> str:
    """Formatta una lista di carte."""
    if not cards:
        return "[]"
    return "[" + ", ".join(format_card(c) for c in cards) + "]"


def print_state(env: ScopaEnv, turn: int) -> None:
    """Stampa lo stato corrente del gioco."""
    print(f"\n{'='*50}")
    print(f"TURNO {turn} | Fase: {env.phase}")
    print(f"{'='*50}")
    print(f"Mazzo: {len(env.engine.deck)} | Tavolo: {format_cards(env.engine.table)}")
    print(f"Mano AI: {format_cards(env.engine.hands[0])} | Mano Avv: {len(env.engine.hands[1])} carte")
    print(f"Prese AI: {len(env.engine.captured[0])} | Prese Avv: {len(env.engine.captured[1])}")
    print(f"Scope AI: {env.engine.scope[0]} | Scope Avv: {env.engine.scope[1]}")
    
    if env.phase == 1 and env.card_played:
        print(f">>> FASE 1: Carta giocata = {format_card(env.card_played)}")
        print(f">>> Selezionate: {format_cards(env.selected_captures)} "
              f"(sum={sum(c.rank for c in env.selected_captures)})")


def play_match() -> None:
    """Esegue una partita visualizzata."""
    env = ScopaEnv()
    obs, info = env.reset()
    
    # Carica modello
    model_path = MODELS_DIR / "scopa_ai_latest.zip"
    model = None
    
    if model_path.exists():
        try:
            from sb3_contrib import MaskablePPO
            model = MaskablePPO.load(str(model_path)[:-4], env=env)
            print("\n" + "=" * 50)
            print("🤖 MODELLO AI CARICATO")
            print("=" * 50)
        except Exception as e:
            print(f"⚠️ Errore caricamento: {e}")
    
    if model is None:
        print("\n" + "=" * 50)
        print("🎲 MODALITÀ RANDOM (modello non trovato)")
        print("=" * 50)
    
    print("\n       NUOVA PARTITA DI SCOPA")
    print("=" * 50)
    
    if info.get("opponent_move"):
        opp_card, opp_taken = info["opponent_move"]
        print("\n⚡ AVVERSARIO INIZIA")
        print(f"   Gioca: {format_card(opp_card)}")
        if opp_taken:
            print(f"   PRENDE: {format_cards(opp_taken)}")
        else:
            print("   (Calata sul tavolo)")
    
    turn = 0
    done = False
    
    while not done:
        turn += 1
        print_state(env, turn)
        
        mask = env.action_masks()
        valid = np.where(mask)[0]
        
        if len(valid) == 0:
            print("ERRORE: Nessuna azione valida!")
            break
        
        if model:
            action, _ = model.predict(obs, action_masks=mask)
        else:
            action = np.random.choice(valid)
        
        # Mostra azione
        if env.phase == 0:
            card = next((c for c in env.engine.hands[0] if c.index == action), None)
            if card:
                print(f"\n-> AI gioca: {format_card(card)}")
        else:
            card = next((c for c in env.engine.table if c.index == action), None)
            if card:
                print(f"\n-> AI seleziona: {format_card(card)}")
        
        obs, reward, done, _, info = env.step(action)
        
        if info.get("ai_move") and info["ai_move"][0]:
            ai_card, ai_taken = info["ai_move"]
            if ai_taken:
                print(f"   Presa: {format_cards(ai_taken)}")
        
        if info.get("opponent_move"):
            opp_card, opp_taken = info["opponent_move"]
            print("\n⚡ AVVERSARIO gioca")
            print(f"   Carta: {format_card(opp_card)}")
            if opp_taken:
                print(f"   PRENDE: {format_cards(opp_taken)}")
            else:
                print("   (Calata)")
        
        if reward != 0:
            print(f"   REWARD: {reward}")
        
        input("\n[INVIO per continuare...]")
    
    # Fine partita
    print("\n" + "=" * 50)
    print("PARTITA TERMINATA!")
    print("=" * 50)
    
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
        print("\n🎉 AI VINCE!")
    elif scores[1] > scores[0]:
        print("\n💀 AVVERSARIO VINCE!")
    else:
        print("\n🤝 PAREGGIO!")


def main():
    """Entry point CLI."""
    play_match()


if __name__ == "__main__":
    main()
