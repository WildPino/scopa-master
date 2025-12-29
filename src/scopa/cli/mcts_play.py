"""
MCTS Play - Partita con IS-MCTS per le decisioni.

Esegue una partita visualizzata dove l'AI usa IS-MCTS
per scegliere le mosse invece di una semplice policy network.

Usage:
    python -m scopa.cli.mcts_play
    python -m scopa.cli.mcts_play --profile tournament
"""
from __future__ import annotations

import argparse
from typing import Optional

import numpy as np

from scopa.game import Card, Suit
from scopa.rl import ScopaEnv
from scopa.search.mcts import ISMCTS, EndgameSolver
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


def play_with_mcts(
    profile: str = "low_latency",
    use_endgame_solver: bool = True,
    verbose: bool = True,
) -> None:
    """
    Esegue una partita usando IS-MCTS per le decisioni.
    
    Args:
        profile: "tournament", "low_latency", o "training"
        use_endgame_solver: se True, usa solver intensivo per endgame
        verbose: stampa dettagli MCTS
    """
    env = ScopaEnv()
    obs, info = env.reset()
    
    # Inizializza MCTS
    mcts = ISMCTS(c_puct=1.0, profile=profile, temperature=0.5)
    endgame_solver = EndgameSolver() if use_endgame_solver else None
    
    print("\n" + "=" * 50)
    print("🎯 PARTITA CON IS-MCTS")
    print(f"   Profilo: {profile} ({mcts.num_det} det × {mcts.sims_per_det} sims)")
    print("=" * 50)
    
    if info.get("opponent_move"):
        opp_card, opp_taken = info["opponent_move"]
        print("\n⚡ AVVERSARIO INIZIA")
        print(f"   Gioca: {format_card(opp_card)}")
        if opp_taken:
            print(f"   PRENDE: {format_cards(opp_taken)}")
    
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
        
        # Check endgame
        if endgame_solver and endgame_solver.is_endgame(env):
            print("\n🎲 ENDGAME - Usando solver intensivo...")
            action = endgame_solver.solve(env)
            action_probs = {action: 1.0}
        else:
            # Usa MCTS
            print(f"\n🧠 MCTS in corso ({mcts.num_det}×{mcts.sims_per_det})...")
            action, action_probs = mcts.search(env, policy_net=None, value_net=None)
        
        # Mostra distribuzione MCTS
        if verbose and len(action_probs) > 0:
            print("   Distribuzione visite:")
            sorted_probs = sorted(action_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            for a, p in sorted_probs:
                card = next((c for c in env.engine.hands[0] if c.index == a), None)
                if card:
                    print(f"     {format_card(card)}: {p:.1%}")
        
        # Mostra azione scelta
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
    
    if scores[0] > scores[1]:
        print("\n🎉 AI VINCE!")
    elif scores[1] > scores[0]:
        print("\n💀 AVVERSARIO VINCE!")
    else:
        print("\n🤝 PAREGGIO!")


def main():
    """Entry point CLI."""
    parser = argparse.ArgumentParser(description="Partita Scopa con IS-MCTS")
    parser.add_argument(
        "--profile", "-p",
        type=str,
        default="low_latency",
        choices=["tournament", "low_latency", "training"],
        help="Profilo MCTS (default: low_latency)"
    )
    parser.add_argument(
        "--no-endgame",
        action="store_true",
        help="Disabilita endgame solver"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Meno output"
    )
    
    args = parser.parse_args()
    
    play_with_mcts(
        profile=args.profile,
        use_endgame_solver=not args.no_endgame,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
