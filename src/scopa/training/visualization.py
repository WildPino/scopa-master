"""
Visualization - Visualizza i progressi dell'addestramento.

Legge i file CSV generati dal Monitor e crea grafici con matplotlib.
Mantiene una memoria storica in JSON per confrontare sessioni diverse.
Ottimizzato per grandi dataset (niente punti raw, solo aggregazioni).

Usage:
    python -m scopa.training.visualization
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from scopa.config import LOGS_DIR, GRAPHS_DIR, ensure_dirs


# Numero di bin per aggregare i dati
N_BINS = 100


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    """Calcola la media mobile per lisciare il grafico."""
    if window_size <= 0 or len(values) < window_size:
        return values
    weights = np.ones(window_size) / window_size
    return np.convolve(values, weights, mode="valid")


def bin_data(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = N_BINS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggrega i dati in bin per una visualizzazione pulita."""
    if len(x) < n_bins:
        return x, y, np.zeros_like(y), np.zeros_like(y), np.zeros_like(y)
    
    bin_edges = np.linspace(x[0], x[-1], n_bins + 1)
    bin_means, bin_stds, bin_p25, bin_p75, bin_centers = [], [], [], [], []
    
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (x >= bin_edges[i]) & (x <= bin_edges[i + 1])
        else:
            mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        
        if np.sum(mask) > 0:
            vals = y[mask]
            bin_means.append(np.mean(vals))
            bin_stds.append(np.std(vals))
            bin_p25.append(np.percentile(vals, 25))
            bin_p75.append(np.percentile(vals, 75))
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
    
    return (
        np.array(bin_centers),
        np.array(bin_means),
        np.array(bin_stds),
        np.array(bin_p25),
        np.array(bin_p75)
    )


def estimate_win_rate(rewards: np.ndarray, threshold: float = 0) -> float:
    """Stima approssimativa del win rate."""
    if len(rewards) == 0:
        return 0.0
    return (np.sum(rewards > threshold) / len(rewards)) * 100


def load_training_history() -> Dict:
    """Carica lo storico delle sessioni di training."""
    history_file = LOGS_DIR / "training_history.json"
    if history_file.exists():
        try:
            return json.loads(history_file.read_text())
        except (json.JSONDecodeError, IOError):
            return {"sessions": []}
    return {"sessions": []}


def save_training_history(history: Dict) -> None:
    """Salva lo storico delle sessioni di training."""
    ensure_dirs()
    history_file = LOGS_DIR / "training_history.json"
    history_file.write_text(json.dumps(history, indent=2))


def backup_old_graphs() -> None:
    """Rinomina i grafici esistenti con suffisso _previous."""
    if not GRAPHS_DIR.exists():
        return
    
    for f in GRAPHS_DIR.glob("*.png"):
        if "_previous" not in f.name:
            new_name = f.name.replace(".png", "_previous.png")
            prev_path = GRAPHS_DIR / new_name
            if prev_path.exists():
                prev_path.unlink()
            shutil.move(f, prev_path)
            print(f"📁 Backup: {f.name} → {new_name}")


def plot_results() -> None:
    """Genera i grafici dall'addestramento."""
    ensure_dirs()
    backup_old_graphs()
    
    try:
        # Cerca tutti i file monitor (sia vecchio formato che nuovo con timestamp)
        import pandas as pd
        from pathlib import Path
        
        monitor_files = list(Path(LOGS_DIR).glob("*.monitor.csv")) + list(Path(LOGS_DIR).glob("monitor.csv"))
        
        if not monitor_files:
            print("❌ Nessun file monitor trovato.")
            return
        
        all_data = []
        for mf in monitor_files:
            try:
                # Leggi il file, skippa l'header JSON (prima riga)
                df = pd.read_csv(mf, skiprows=1)
                if len(df) > 0:
                    all_data.append(df)
            except Exception as e:
                print(f"⚠️ Errore leggendo {mf.name}: {e}")
        
        if not all_data:
            print("❌ Nessun dato valido nei file monitor.")
            return
        
        # Unisci tutti i dati
        results = pd.concat(all_data, ignore_index=True)
        
        # Calcola timesteps cumulativi
        episode_lengths = results["l"].values
        x = np.cumsum(episode_lengths)
        y = results["r"].values
    except Exception as e:
        print(f"❌ Errore nel caricamento dei log: {e}")
        return
    
    if len(x) == 0:
        print("❌ Nessun dato trovato nei log.")
        return
    
    print(f"📊 Trovati {len(x):,} episodi da {len(monitor_files)} file monitor")
    print(f"   Timesteps totali: {x[-1]:,}")
    print(f"   Aggregazione in {N_BINS} bin per grafici leggibili")
    
    # Aggiorna storico
    history = load_training_history()
    session_data = {
        "timestamp": datetime.now().isoformat(),
        "episodes": int(len(y)),
        "total_timesteps": int(x[-1]) if len(x) > 0 else 0,
        "mean_reward": float(np.mean(y)),
        "max_reward": float(np.max(y)),
        "min_reward": float(np.min(y)),
        "std_reward": float(np.std(y)),
        "win_rate_approx": float(estimate_win_rate(y)),
        "mean_episode_length": float(np.mean(episode_lengths)) if len(episode_lengths) > 0 else 0,
    }
    history["sessions"].append(session_data)
    save_training_history(history)
    print(f"💾 Sessione salvata ({len(history['sessions'])} totali)")
    
    # Aggregazione
    x_bin, y_mean, y_std, y_p25, y_p75 = bin_data(x, y)
    
    if len(episode_lengths) > 0:
        _, len_mean, len_std, _, _ = bin_data(x, episode_lengths)
    
    # Calcola win rate per bin
    bin_edges = np.linspace(x[0], x[-1], N_BINS + 1)
    win_rates = []
    for i in range(N_BINS):
        if i == N_BINS - 1:
            mask = (x >= bin_edges[i]) & (x <= bin_edges[i + 1])
        else:
            mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        if np.sum(mask) > 0:
            win_rates.append(estimate_win_rate(y[mask]))
    win_rates = np.array(win_rates) if win_rates else np.array([50])
    
    # === FIGURA 1: REWARD ANALYSIS (3 grafici) ===
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle(f"📈 Analisi Reward ({len(y):,} episodi → {N_BINS} bin)", fontsize=12, fontweight="bold")
    
    # 1.1 Reward con banda di confidenza
    ax = axes1[0]
    ax.fill_between(x_bin, y_p25, y_p75, alpha=0.3, color="blue", label="25°-75° percentile")
    ax.plot(x_bin, y_mean, color="darkblue", linewidth=2, label="Media")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.set_title("Reward nel Tempo")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 1.2 Win Rate
    ax = axes1[1]
    ax.plot(x_bin[:len(win_rates)], win_rates, color="green", linewidth=2)
    ax.axhline(50, color="red", linestyle="--", alpha=0.7, label="50%")
    ax.fill_between(x_bin[:len(win_rates)], 50, win_rates, where=(win_rates >= 50), 
                    alpha=0.3, color="green", interpolate=True)
    ax.fill_between(x_bin[:len(win_rates)], 50, win_rates, where=(win_rates < 50), 
                    alpha=0.3, color="red", interpolate=True)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Win Rate %")
    ax.set_title("Win Rate nel Tempo")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 1.3 Distribuzione Reward
    ax = axes1[2]
    ax.hist(y, bins=50, edgecolor="black", alpha=0.7, color="steelblue", density=True)
    ax.axvline(np.mean(y), color="red", linestyle="-", linewidth=2, label=f"Media: {np.mean(y):.2f}")
    ax.axvline(np.median(y), color="orange", linestyle="--", linewidth=2, label=f"Mediana: {np.median(y):.2f}")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Reward")
    ax.set_ylabel("Densità")
    ax.set_title("Distribuzione Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1.savefig(GRAPHS_DIR / "reward_analysis.png", dpi=150)
    plt.close(fig1)
    
    # === FIGURA 2: PERFORMANCE (3 grafici) ===
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle("📊 Analisi Performance", fontsize=12, fontweight="bold")
    
    # 2.1 Episode Length
    ax = axes2[0]
    if len(episode_lengths) > 0:
        ax.plot(x_bin, len_mean, color="teal", linewidth=2, label="Media")
        ax.fill_between(x_bin, len_mean - len_std, len_mean + len_std, 
                        alpha=0.3, color="teal", label="±1 std")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Lunghezza Episodio")
        ax.set_title("Lunghezza Episodi")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Dati non disponibili", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, alpha=0.3)
    
    # 2.2 Cumulative Reward
    ax = axes2[1]
    cumulative = np.cumsum(y)
    step = max(1, len(cumulative) // 500)
    ax.plot(x[::step], cumulative[::step], color="darkblue", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(x[::step], 0, cumulative[::step], 
                    where=(cumulative[::step] >= 0), alpha=0.2, color="green")
    ax.fill_between(x[::step], 0, cumulative[::step], 
                    where=(cumulative[::step] < 0), alpha=0.2, color="red")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward Cumulativo")
    ax.set_title(f"Reward Cumulativo: {cumulative[-1]:.0f}")
    ax.grid(True, alpha=0.3)
    
    # 2.3 Heatmap Reward vs Episode Length
    ax = axes2[2]
    if len(episode_lengths) > 0:
        h, xedges, yedges = np.histogram2d(episode_lengths, y, bins=30)
        im = ax.imshow(h.T, origin="lower", aspect="auto", cmap="viridis",
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.colorbar(im, ax=ax, label="Frequenza")
        ax.set_xlabel("Lunghezza Episodio")
        ax.set_ylabel("Reward")
        ax.set_title("Heatmap Length vs Reward")
    else:
        ax.text(0.5, 0.5, "Dati non disponibili", ha="center", va="center", transform=ax.transAxes)
    
    plt.tight_layout()
    fig2.savefig(GRAPHS_DIR / "performance_analysis.png", dpi=150)
    plt.close(fig2)
    
    # === FIGURA VALUE LOSS (SINGOLA) ===
    # Leggi il CSV della value_loss se esiste
    value_loss_file = LOGS_DIR / "value_loss.csv"
    if value_loss_file.exists():
        try:
            import pandas as pd
            df_vl = pd.read_csv(value_loss_file)
            
            if len(df_vl) > 10:  # Servono abbastanza dati
                fig_vl, ax_vl = plt.subplots(figsize=(12, 6))
                
                timesteps_vl = df_vl["timesteps"].values
                value_loss = df_vl["value_loss"].values
                
                # Aggregazione in bin per visualizzazione pulita
                n_bins_vl = min(100, len(value_loss) // 5 + 1)
                if n_bins_vl > 2:
                    x_bin_vl, y_mean_vl, y_std_vl, y_p25_vl, y_p75_vl = bin_data(
                        timesteps_vl, value_loss, n_bins_vl
                    )
                else:
                    x_bin_vl = timesteps_vl
                    y_mean_vl = value_loss
                    y_std_vl = np.zeros_like(value_loss)
                    y_p25_vl = value_loss
                    y_p75_vl = value_loss
                
                # Grafico principale con banda di confidenza
                ax_vl.fill_between(
                    x_bin_vl, y_p25_vl, y_p75_vl, 
                    alpha=0.2, color="red", label="25°-75° percentile"
                )
                ax_vl.plot(x_bin_vl, y_mean_vl, color="darkred", linewidth=2, label="Value Loss Media")
                
                # Media mobile per vedere il trend
                if len(y_mean_vl) > 10:
                    ma = moving_average(y_mean_vl, min(20, len(y_mean_vl) // 5))
                    ma_x = x_bin_vl[len(x_bin_vl) - len(ma):]
                    ax_vl.plot(ma_x, ma, color="blue", linewidth=2.5, linestyle="--", 
                              alpha=0.8, label="Media Mobile (trend)")
                
                # Linea target (value_loss ideale)
                ax_vl.axhline(1.0, color="green", linestyle=":", alpha=0.7, linewidth=2, 
                             label="Target ideale (~1.0)")
                
                # Annotazione per indicare la convergenza
                current_vl = y_mean_vl[-1]
                initial_vl = y_mean_vl[0]
                reduction = (initial_vl - current_vl) / initial_vl * 100 if initial_vl > 0 else 0
                
                convergence_emoji = "✅" if current_vl < 3.0 else "🔄" if current_vl < 5.0 else "⚠️"
                ax_vl.annotate(
                    f"{convergence_emoji} Attuale: {current_vl:.2f}\nRiduzione: {reduction:.1f}%",
                    xy=(timesteps_vl[-1], current_vl),
                    xytext=(0.02, 0.15), textcoords="axes fraction",
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"),
                    ha="left"
                )
                
                ax_vl.set_xlabel("Timesteps", fontsize=11)
                ax_vl.set_ylabel("Value Loss", fontsize=11)
                ax_vl.set_title("📉 Value Loss Convergence", fontsize=12, fontweight="bold")
                ax_vl.legend(loc="upper right", fontsize=9)
                ax_vl.grid(True, alpha=0.3)
                ax_vl.set_ylim(bottom=0)  # Non mostrare valori negativi
                
                plt.tight_layout()
                fig_vl.savefig(GRAPHS_DIR / "value_loss.png", dpi=150)
                plt.close(fig_vl)
                print("✅ Grafico Value Loss salvato!")
        except Exception as e:
            print(f"⚠️ Impossibile creare grafico Value Loss: {e}")
    else:
        print("ℹ️ Nessun dato value_loss.csv trovato - avvia un training per generarlo")
    
    
    # === FIGURA 3: STORICO SESSIONI ===
    if len(history["sessions"]) > 1:
        sessions = history["sessions"]
        n_sessions = len(sessions)
        
        fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
        fig3.suptitle(f"📅 Storico: {n_sessions} Sessioni di Training", fontsize=12, fontweight="bold")
        
        session_nums = list(range(1, n_sessions + 1))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_sessions))
        
        # 3.1 Reward medio
        ax = axes3[0, 0]
        means = [s["mean_reward"] for s in sessions]
        ax.bar(session_nums, means, color=colors, edgecolor="black", alpha=0.8)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(np.mean(means), color="red", linestyle="-", alpha=0.7, 
                   label=f"Media globale: {np.mean(means):.2f}")
        ax.set_xlabel("Sessione")
        ax.set_ylabel("Reward Medio")
        ax.set_title("Reward Medio per Sessione")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        
        # 3.2 Win Rate
        ax = axes3[0, 1]
        win_rates_hist = [s.get("win_rate_approx", 0) for s in sessions]
        ax.bar(session_nums, win_rates_hist, color=colors, edgecolor="black", alpha=0.8)
        ax.axhline(50, color="red", linestyle="--", alpha=0.7, label="50%")
        ax.set_xlabel("Sessione")
        ax.set_ylabel("Win Rate %")
        ax.set_title("Win Rate per Sessione")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        
        # 3.3 Trend complessivo
        ax = axes3[1, 0]
        cumulative_ts = np.cumsum([s.get("total_timesteps", 0) for s in sessions])
        ax.plot(session_nums, means, "o-", color="steelblue", linewidth=2, markersize=8, label="Reward")
        
        if n_sessions >= 3:
            z = np.polyfit(session_nums, means, 1)
            p = np.poly1d(z)
            ax.plot(session_nums, p(session_nums), "--", color="red", alpha=0.7, 
                    label=f"Trend: {z[0]:+.3f}/sessione")
        
        ax.set_xlabel("Sessione")
        ax.set_ylabel("Reward Medio")
        ax.set_title("Evoluzione nel Tempo")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 3.4 Timesteps cumulativi
        ax = axes3[1, 1]
        ax.bar(session_nums, [s.get("total_timesteps", 0) for s in sessions], 
               color=colors, edgecolor="black", alpha=0.6, label="Per sessione")
        ax.plot(session_nums, cumulative_ts, "ro-", linewidth=2, markersize=6, label="Cumulativo")
        ax.set_xlabel("Sessione")
        ax.set_ylabel("Timesteps")
        ax.set_title(f"Timesteps Totali: {cumulative_ts[-1]:,}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        fig3.savefig(GRAPHS_DIR / "session_history.png", dpi=150)
        plt.close(fig3)
        print("✅ Grafico storico sessioni salvato!")
    
    print(f"\n✅ Grafici salvati in: {GRAPHS_DIR}")
    
    # Statistiche
    print(f"\n{'='*50}")
    print("📈 STATISTICHE SESSIONE CORRENTE")
    print(f"{'='*50}")
    print(f"   Episodi: {len(y):,}")
    print(f"   Timesteps: {x[-1]:,}")
    print(f"   Reward medio: {np.mean(y):.3f}")
    print(f"   Reward mediana: {np.median(y):.3f}")
    print(f"   Reward range: [{np.min(y):.1f}, {np.max(y):.1f}]")
    print(f"   Win rate: {estimate_win_rate(y):.1f}%")
    if len(episode_lengths) > 0:
        print(f"   Ep. length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    # Trend
    if len(y) >= 100:
        n = len(y) // 4
        early = np.mean(y[:n])
        late = np.mean(y[-n:])
        delta = late - early
        emoji = "📈" if delta > 0.1 else "📉" if delta < -0.1 else "➡️"
        print(f"   {emoji} Trend: {early:.2f} → {late:.2f} ({delta:+.2f})")


def main():
    """Entry point CLI."""
    plot_results()


if __name__ == "__main__":
    main()
