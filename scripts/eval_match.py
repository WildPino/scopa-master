"""
Evaluation Match Script - Valutazione modelli Scopa AI.

Esegue N partite contro un pool di avversari e calcola metriche:
- Winrate (totale e per ruolo primo/secondo)
- Point differential medio
- Statistiche dettagliate (scope, settebello, denari, primiera)

Usage:
    python scripts/eval_match.py --model models/scopa_ai_latest.zip --opponents random heuristic --games 100
    python scripts/eval_match.py --model models/scopa_ai_latest.zip --opponents random --games 50 --output results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from scopa.rl import ScopaEnv
from scopa.game import Suit
from scopa.config import MODELS_DIR


@dataclass
class GameResult:
    """Risultato di una singola partita."""
    winner: int  # 0=AI, 1=opponent, -1=draw
    score_ai: int
    score_opponent: int
    scope_ai: int
    scope_opponent: int
    cards_ai: int
    cards_opponent: int
    denari_ai: int
    denari_opponent: int
    settebello_ai: bool
    primiera_ai: int
    primiera_opponent: int
    ai_started: bool
    num_turns: int


@dataclass 
class EvalResults:
    """Risultati aggregati della valutazione."""
    model_path: str
    opponent_type: str
    total_games: int
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_score_ai: int = 0
    total_score_opponent: int = 0
    wins_as_first: int = 0
    games_as_first: int = 0
    wins_as_second: int = 0
    games_as_second: int = 0
    total_scope_ai: int = 0
    total_settebello_ai: int = 0
    total_denari_ai: int = 0
    total_cards_ai: int = 0
    games_results: List[GameResult] = field(default_factory=list)
    
    @property
    def winrate(self) -> float:
        return self.wins / self.total_games if self.total_games > 0 else 0.0
    
    @property
    def winrate_as_first(self) -> float:
        return self.wins_as_first / self.games_as_first if self.games_as_first > 0 else 0.0
    
    @property
    def winrate_as_second(self) -> float:
        return self.wins_as_second / self.games_as_second if self.games_as_second > 0 else 0.0
    
    @property
    def avg_point_diff(self) -> float:
        return (self.total_score_ai - self.total_score_opponent) / self.total_games if self.total_games > 0 else 0.0
    
    @property
    def avg_scope_per_game(self) -> float:
        return self.total_scope_ai / self.total_games if self.total_games > 0 else 0.0
    
    @property
    def settebello_rate(self) -> float:
        return self.total_settebello_ai / self.total_games if self.total_games > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "opponent_type": self.opponent_type,
            "total_games": self.total_games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "winrate": round(self.winrate, 4),
            "winrate_as_first": round(self.winrate_as_first, 4),
            "winrate_as_second": round(self.winrate_as_second, 4),
            "games_as_first": self.games_as_first,
            "games_as_second": self.games_as_second,
            "avg_point_diff": round(self.avg_point_diff, 2),
            "avg_scope_per_game": round(self.avg_scope_per_game, 2),
            "settebello_rate": round(self.settebello_rate, 4),
            "total_score_ai": self.total_score_ai,
            "total_score_opponent": self.total_score_opponent,
        }


def load_model(model_path: str) -> Optional[Any]:
    """Carica un modello MaskablePPO."""
    try:
        from sb3_contrib import MaskablePPO
        # Remove .zip extension if present
        path = model_path[:-4] if model_path.endswith(".zip") else model_path
        model = MaskablePPO.load(path)
        return model
    except Exception as e:
        print(f"⚠️ Errore caricamento modello: {e}")
        return None


def play_single_game(
    env: ScopaEnv,
    model: Optional[Any],
    verbose: bool = False
) -> GameResult:
    """Esegue una singola partita e ritorna il risultato."""
    obs, info = env.reset()
    
    # Determina chi inizia
    ai_started = env.starter_player == 0
    
    done = False
    num_turns = 0
    
    while not done:
        num_turns += 1
        mask = env.action_masks()
        
        if model is not None:
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        else:
            # Random fallback
            valid = np.where(mask)[0]
            action = np.random.choice(valid)
        
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
    
    # Calcola statistiche finali
    scores = env.engine.calculate_score()
    captured_ai = env.engine.captured[0]
    captured_opp = env.engine.captured[1]
    
    denari_ai = sum(1 for c in captured_ai if c.suit == Suit.DENARI)
    denari_opp = sum(1 for c in captured_opp if c.suit == Suit.DENARI)
    
    settebello_ai = any(c.rank == 7 and c.suit == Suit.DENARI for c in captured_ai)
    
    primiera_ai = env.engine._get_primiera_score(captured_ai)
    primiera_opp = env.engine._get_primiera_score(captured_opp)
    
    # Determina vincitore
    if scores[0] > scores[1]:
        winner = 0
    elif scores[1] > scores[0]:
        winner = 1
    else:
        winner = -1
    
    return GameResult(
        winner=winner,
        score_ai=scores[0],
        score_opponent=scores[1],
        scope_ai=env.engine.scope[0],
        scope_opponent=env.engine.scope[1],
        cards_ai=len(captured_ai),
        cards_opponent=len(captured_opp),
        denari_ai=denari_ai,
        denari_opponent=denari_opp,
        settebello_ai=settebello_ai,
        primiera_ai=primiera_ai,
        primiera_opponent=primiera_opp,
        ai_started=ai_started,
        num_turns=num_turns,
    )


def evaluate_model(
    model_path: str,
    opponent_type: str,
    num_games: int,
    verbose: bool = False
) -> EvalResults:
    """Valuta un modello contro un tipo di avversario."""
    
    # Carica modello
    model = None
    if model_path and model_path != "random":
        model = load_model(model_path)
        if model is None and model_path != "random":
            print(f"⚠️ Modello non caricato, uso random")
    
    # Crea ambiente
    env = ScopaEnv(opponent_mode=opponent_type)
    
    results = EvalResults(
        model_path=model_path or "random",
        opponent_type=opponent_type,
        total_games=num_games
    )
    
    print(f"\n🎮 Valutazione: {results.model_path} vs {opponent_type} ({num_games} partite)")
    print("-" * 60)
    
    start_time = time.time()
    
    for i in range(num_games):
        game_result = play_single_game(env, model, verbose=verbose)
        results.games_results.append(game_result)
        
        # Aggiorna statistiche aggregate
        if game_result.winner == 0:
            results.wins += 1
        elif game_result.winner == 1:
            results.losses += 1
        else:
            results.draws += 1
        
        results.total_score_ai += game_result.score_ai
        results.total_score_opponent += game_result.score_opponent
        results.total_scope_ai += game_result.scope_ai
        results.total_denari_ai += game_result.denari_ai
        results.total_cards_ai += game_result.cards_ai
        
        if game_result.settebello_ai:
            results.total_settebello_ai += 1
        
        if game_result.ai_started:
            results.games_as_first += 1
            if game_result.winner == 0:
                results.wins_as_first += 1
        else:
            results.games_as_second += 1
            if game_result.winner == 0:
                results.wins_as_second += 1
        
        # Progress
        if (i + 1) % max(1, num_games // 10) == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{num_games} partite | WR: {results.winrate:.1%} | {rate:.1f} games/s")
    
    elapsed = time.time() - start_time
    env.close()
    
    print("-" * 60)
    print(f"✅ Completato in {elapsed:.1f}s ({num_games/elapsed:.1f} games/s)")
    
    return results


def print_results(results: EvalResults) -> None:
    """Stampa risultati formattati."""
    print(f"\n{'='*60}")
    print(f"📊 RISULTATI: {results.model_path} vs {results.opponent_type}")
    print(f"{'='*60}")
    print(f"  Partite totali: {results.total_games}")
    print(f"  Vittorie: {results.wins} | Sconfitte: {results.losses} | Pareggi: {results.draws}")
    print(f"")
    print(f"  📈 WINRATE: {results.winrate:.1%}")
    print(f"     - Come primo: {results.winrate_as_first:.1%} ({results.wins_as_first}/{results.games_as_first})")
    print(f"     - Come secondo: {results.winrate_as_second:.1%} ({results.wins_as_second}/{results.games_as_second})")
    print(f"")
    print(f"  📊 STATISTICHE:")
    print(f"     - Point diff medio: {results.avg_point_diff:+.2f}")
    print(f"     - Scope per partita: {results.avg_scope_per_game:.2f}")
    print(f"     - Settebello rate: {results.settebello_rate:.1%}")
    print(f"{'='*60}\n")


def save_results(
    all_results: List[EvalResults],
    output_path: str,
    format: str = "json"
) -> None:
    """Salva risultati su file."""
    path = Path(output_path)
    
    if format == "json":
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [r.to_dict() for r in all_results]
        }
        path.write_text(json.dumps(data, indent=2))
    else:
        # CSV format
        import csv
        with open(path, 'w', newline='') as f:
            if all_results:
                writer = csv.DictWriter(f, fieldnames=all_results[0].to_dict().keys())
                writer.writeheader()
                for r in all_results:
                    writer.writerow(r.to_dict())
    
    print(f"💾 Risultati salvati in: {path}")


def main():
    """Entry point CLI."""
    parser = argparse.ArgumentParser(
        description="Valuta un modello Scopa AI contro diversi avversari"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path al modello da valutare (default: random)"
    )
    parser.add_argument(
        "--opponents", "-o",
        nargs="+",
        default=["random", "heuristic"],
        choices=["random", "heuristic", "self"],
        help="Tipi di avversari da testare"
    )
    parser.add_argument(
        "--games", "-n",
        type=int,
        default=100,
        help="Numero di partite per avversario"
    )
    parser.add_argument(
        "--output", "-f",
        type=str,
        default=None,
        help="File output per risultati (JSON o CSV)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Output dettagliato"
    )
    
    args = parser.parse_args()
    
    # Risolvi path modello
    model_path = args.model
    if model_path and not Path(model_path).exists():
        # Prova nella directory models
        alt_path = MODELS_DIR / model_path
        if alt_path.exists():
            model_path = str(alt_path)
        elif (MODELS_DIR / f"{model_path}.zip").exists():
            model_path = str(MODELS_DIR / f"{model_path}.zip")
    
    all_results = []
    
    for opponent in args.opponents:
        results = evaluate_model(
            model_path=model_path,
            opponent_type=opponent,
            num_games=args.games,
            verbose=args.verbose
        )
        print_results(results)
        all_results.append(results)
    
    # Salva risultati
    if args.output:
        fmt = "csv" if args.output.endswith(".csv") else "json"
        save_results(all_results, args.output, format=fmt)
    
    # Summary finale
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("📋 RIEPILOGO FINALE")
        print("="*60)
        for r in all_results:
            print(f"  vs {r.opponent_type:12} | WR: {r.winrate:.1%} | Δpts: {r.avg_point_diff:+.2f}")
        print("="*60)


if __name__ == "__main__":
    main()
