"""
Visualization - Visualizza i progressi dell'addestramento.

Supporta due formati di log:
1. Monitor CSV (SB3): logs/*.monitor.csv
2. Train Recurrent CSV: logs/train_recurrent/*.csv

Usage:
    python -m scopa.training.visualization
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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


def load_recurrent_logs() -> Optional[Dict[str, np.ndarray]]:
    """
    Carica i log da train_recurrent.py.
    
    Returns:
        Dict con arrays per ogni metrica, o None se non trovato
    """
    import pandas as pd
    
    log_dir = LOGS_DIR / "train_recurrent"
    if not log_dir.exists():
        return None
    
    csv_files = sorted(log_dir.glob("*.csv"))
    if not csv_files:
        return None
    
    all_data = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                all_data.append(df)
        except Exception as e:
            print(f"⚠️ Errore leggendo {f.name}: {e}")
    
    if not all_data:
        return None
    
    # Unisci tutti i dati
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values("timesteps").drop_duplicates(subset=["timesteps"], keep="last")
    
    return {
        "timesteps": df["timesteps"].values,
        "time_elapsed": df["time_elapsed"].values if "time_elapsed" in df else None,
        "fps": df["fps"].values if "fps" in df else None,
        "policy_loss": df["policy_loss"].values if "policy_loss" in df else None,
        "value_loss": df["value_loss"].values if "value_loss" in df else None,
        "entropy": df["entropy"].values if "entropy" in df else None,
        "lr": df["lr"].values if "lr" in df else None,
        "ent_coef": df["ent_coef"].values if "ent_coef" in df else None,
        "wr_random": df["wr_random"].values if "wr_random" in df else None,
        "wr_heuristic": df["wr_heuristic"].values if "wr_heuristic" in df else None,
    }


def load_sessions_data() -> Optional[List[Dict]]:
    """
    Carica i log separati per sessione di training.
    
    Returns:
        Lista di dict, uno per ogni sessione con metadata e dati
    """
    import pandas as pd
    from datetime import datetime
    
    log_dir = LOGS_DIR / "train_recurrent"
    if not log_dir.exists():
        return None
    
    csv_files = sorted(log_dir.glob("*.csv"))
    if not csv_files:
        return None
    
    sessions = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if len(df) == 0:
                continue
            
            # Estrai info dalla sessione dal nome file
            # Formato: name_YYYYMMDD_HHMMSS.csv
            name_parts = f.stem.split("_")
            if len(name_parts) >= 3:
                date_str = name_parts[-2]
                time_str = name_parts[-1]
                try:
                    session_dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                    session_label = session_dt.strftime("%d/%m %H:%M")
                except ValueError:
                    session_label = f.stem[-15:]
            else:
                session_label = f.stem
            
            sessions.append({
                "filename": f.name,
                "label": session_label,
                "start_timesteps": df["timesteps"].iloc[0],
                "end_timesteps": df["timesteps"].iloc[-1],
                "duration_steps": df["timesteps"].iloc[-1] - df["timesteps"].iloc[0],
                "n_evals": len(df),
                "final_wr_random": df["wr_random"].iloc[-1] if "wr_random" in df else None,
                "final_wr_heuristic": df["wr_heuristic"].iloc[-1] if "wr_heuristic" in df else None,
                "best_wr_random": df["wr_random"].max() if "wr_random" in df else None,
                "best_wr_heuristic": df["wr_heuristic"].max() if "wr_heuristic" in df else None,
                "final_value_loss": df["value_loss"].iloc[-1] if "value_loss" in df else None,
                "final_entropy": df["entropy"].iloc[-1] if "entropy" in df else None,
                "data": df,
            })
        except Exception as e:
            print(f"⚠️ Errore leggendo {f.name}: {e}")
    
    return sessions if sessions else None


def load_monitor_logs() -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Carica i log dal Monitor di SB3 (formato legacy).
    
    Returns:
        (timesteps, rewards, episode_lengths) o None se non trovato
    """
    import pandas as pd
    
    monitor_files = list(Path(LOGS_DIR).glob("*.monitor.csv")) + list(Path(LOGS_DIR).glob("monitor.csv"))
    
    if not monitor_files:
        return None
    
    all_data = []
    for mf in monitor_files:
        try:
            df = pd.read_csv(mf, skiprows=1)
            if len(df) > 0:
                all_data.append(df)
        except Exception:
            pass
    
    if not all_data:
        return None
    
    results = pd.concat(all_data, ignore_index=True)
    
    episode_lengths = results["l"].values
    x = np.cumsum(episode_lengths)
    y = results["r"].values
    
    return x, y, episode_lengths


def plot_recurrent_results(data: Dict[str, np.ndarray]) -> None:
    """Genera grafici per train_recurrent.py logs."""
    ensure_dirs()
    backup_old_graphs()
    
    timesteps = data["timesteps"]
    n_points = len(timesteps)
    
    print(f"📊 Trovati {n_points} punti di log")
    print(f"   Timesteps: {timesteps[0]:,} → {timesteps[-1]:,}")
    
    # === FIGURA 1: Training Metrics (2x2) ===
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle(f"📈 Training Metrics ({timesteps[-1]:,} timesteps)", fontsize=14, fontweight="bold")
    
    # 1.1 Win Rates
    ax = axes1[0, 0]
    if data["wr_random"] is not None:
        wr_random = data["wr_random"] * 100
        wr_heuristic = data["wr_heuristic"] * 100 if data["wr_heuristic"] is not None else None
        
        ax.plot(timesteps, wr_random, 'b-', linewidth=2, label="vs Random", marker='o', markersize=4)
        if wr_heuristic is not None:
            ax.plot(timesteps, wr_heuristic, 'g-', linewidth=2, label="vs Heuristic", marker='s', markersize=4)
        
        ax.axhline(50, color="red", linestyle="--", alpha=0.7, label="50% baseline")
        ax.fill_between(timesteps, 50, wr_random, where=(wr_random >= 50), 
                        alpha=0.2, color="blue", interpolate=True)
        ax.fill_between(timesteps, 50, wr_random, where=(wr_random < 50), 
                        alpha=0.2, color="red", interpolate=True)
        
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Win Rate %")
        ax.set_title("🎯 Win Rate vs Opponents")
        ax.set_ylim(0, 100)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        # Annotazione ultimo valore
        ax.annotate(f"{wr_random[-1]:.1f}%", (timesteps[-1], wr_random[-1]),
                   textcoords="offset points", xytext=(5, 5), fontsize=9, fontweight="bold", color="blue")
        if wr_heuristic is not None:
            ax.annotate(f"{wr_heuristic[-1]:.1f}%", (timesteps[-1], wr_heuristic[-1]),
                       textcoords="offset points", xytext=(5, -10), fontsize=9, fontweight="bold", color="green")
    else:
        ax.text(0.5, 0.5, "Win rate data not available", ha="center", va="center", transform=ax.transAxes)
    
    # 1.2 Losses
    ax = axes1[0, 1]
    if data["policy_loss"] is not None and data["value_loss"] is not None:
        ax2 = ax.twinx()
        
        l1 = ax.plot(timesteps, data["policy_loss"], 'b-', linewidth=2, label="Policy Loss")
        l2 = ax2.plot(timesteps, data["value_loss"], 'r-', linewidth=2, label="Value Loss")
        
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Policy Loss", color="blue")
        ax2.set_ylabel("Value Loss", color="red")
        ax.tick_params(axis='y', labelcolor="blue")
        ax2.tick_params(axis='y', labelcolor="red")
        ax.set_title("📉 Training Losses")
        
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper right")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Loss data not available", ha="center", va="center", transform=ax.transAxes)
    
    # 1.3 Entropy
    ax = axes1[1, 0]
    if data["entropy"] is not None:
        ax.plot(timesteps, data["entropy"], 'purple', linewidth=2)
        ax.fill_between(timesteps, 0, data["entropy"], alpha=0.3, color="purple")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Entropy")
        ax.set_title("🎲 Policy Entropy")
        ax.grid(True, alpha=0.3)
        
        # Trend line
        if len(timesteps) >= 3:
            z = np.polyfit(timesteps, data["entropy"], 1)
            trend = np.poly1d(z)(timesteps)
            ax.plot(timesteps, trend, 'r--', linewidth=1.5, alpha=0.7, label="Trend")
            ax.legend()
    else:
        ax.text(0.5, 0.5, "Entropy data not available", ha="center", va="center", transform=ax.transAxes)
    
    # 1.4 Learning Rate & Entropy Coef
    ax = axes1[1, 1]
    if data["lr"] is not None:
        ax2 = ax.twinx()
        
        l1 = ax.plot(timesteps, data["lr"], 'g-', linewidth=2, label="Learning Rate")
        if data["ent_coef"] is not None:
            l2 = ax2.plot(timesteps, data["ent_coef"], 'm-', linewidth=2, label="Entropy Coef")
        else:
            l2 = []
        
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Learning Rate", color="green")
        ax2.set_ylabel("Entropy Coef", color="magenta")
        ax.tick_params(axis='y', labelcolor="green")
        ax2.tick_params(axis='y', labelcolor="magenta")
        ax.set_title("⚙️ Hyperparameters Schedule")
        
        lines = l1 + (l2 if l2 else [])
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper right")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Hyperparameter data not available", ha="center", va="center", transform=ax.transAxes)
    
    plt.tight_layout()
    fig1.savefig(GRAPHS_DIR / "training_metrics.png", dpi=150)
    plt.close(fig1)
    print("✅ Salvato: training_metrics.png")
    
    # === FIGURA 2: Value Loss Convergence (singola) ===
    if data["value_loss"] is not None and len(data["value_loss"]) > 3:
        fig2, ax = plt.subplots(figsize=(12, 6))
        
        vl = data["value_loss"]
        
        # Linea principale
        ax.plot(timesteps, vl, 'darkred', linewidth=2, label="Value Loss")
        
        # Media mobile
        if len(vl) > 5:
            window = max(3, len(vl) // 10)
            ma = moving_average(vl, window)
            ma_x = timesteps[len(timesteps) - len(ma):]
            ax.plot(ma_x, ma, 'blue', linewidth=2.5, linestyle="--", alpha=0.8, label="Media Mobile")
        
        # Target ideale
        ax.axhline(1.0, color="green", linestyle=":", alpha=0.7, linewidth=2, label="Target ~1.0")
        
        # Annotazioni
        current_vl = vl[-1]
        initial_vl = vl[0]
        reduction = (initial_vl - current_vl) / initial_vl * 100 if initial_vl > 0 else 0
        
        emoji = "✅" if current_vl < 3.0 else "🔄" if current_vl < 5.0 else "⚠️"
        ax.annotate(
            f"{emoji} Attuale: {current_vl:.2f}\nRiduzione: {reduction:.1f}%",
            xy=(timesteps[-1], current_vl),
            xytext=(0.02, 0.85), textcoords="axes fraction",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"),
            ha="left"
        )
        
        ax.set_xlabel("Timesteps", fontsize=12)
        ax.set_ylabel("Value Loss", fontsize=12)
        ax.set_title("📉 Value Loss Convergence", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        fig2.savefig(GRAPHS_DIR / "value_loss.png", dpi=150)
        plt.close(fig2)
        print("✅ Salvato: value_loss.png")
    
    # === FIGURA 3: AI Evolution - Grafico Unificato ===
    sessions = load_sessions_data()
    if sessions and len(sessions) >= 1:
        # Crea figura con 4 subplot per una vista completa
        fig3, axes3 = plt.subplots(4, 1, figsize=(16, 14), height_ratios=[2, 1, 1, 1])
        fig3.suptitle(f"AI Evolution - Training Timeline ({len(sessions)} sessioni, {timesteps[-1]:,} steps)", 
                     fontsize=14, fontweight="bold")
        
        # Colori sessioni (palette distinta)
        session_colors = plt.cm.Set1(np.linspace(0, 1, max(len(sessions), 3)))
        
        # Calcola i boundaries delle sessioni
        session_boundaries = []
        for s in sessions:
            session_boundaries.append({
                "start": s["start_timesteps"],
                "end": s["end_timesteps"],
                "label": s["label"]
            })
        
        def add_session_bands(ax, alpha=0.1):
            """Aggiunge bande colorate per ogni sessione."""
            for i, bound in enumerate(session_boundaries):
                ax.axvspan(bound["start"], bound["end"], 
                          alpha=alpha, color=session_colors[i], zorder=0)
                # Label sessione in alto
                mid = (bound["start"] + bound["end"]) / 2
                ax.annotate(f"S{i+1}", xy=(mid, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1), 
                           fontsize=8, ha="center", va="bottom", alpha=0.7)
        
        # === SUBPLOT 1: Win Rates (principale) ===
        ax1 = axes3[0]
        
        # Plot win rates come linea continua
        if data["wr_random"] is not None:
            wr_r = data["wr_random"] * 100
            wr_h = data["wr_heuristic"] * 100 if data["wr_heuristic"] is not None else None
            
            ax1.plot(timesteps, wr_r, 'b-', linewidth=2.5, label="vs Random", marker='o', markersize=4)
            if wr_h is not None:
                ax1.plot(timesteps, wr_h, 'g-', linewidth=2.5, label="vs Heuristic", marker='s', markersize=4)
            
            # Media mobile per trend
            if len(wr_r) > 5:
                window = max(3, len(wr_r) // 5)
                ma_r = moving_average(wr_r, window)
                ma_x = timesteps[len(timesteps) - len(ma_r):]
                ax1.plot(ma_x, ma_r, 'b--', linewidth=1.5, alpha=0.5, label="Trend Random")
                if wr_h is not None:
                    ma_h = moving_average(wr_h, window)
                    ax1.plot(ma_x, ma_h, 'g--', linewidth=1.5, alpha=0.5, label="Trend Heuristic")
            
            # Fill area sopra/sotto 50%
            ax1.fill_between(timesteps, 50, wr_r, where=(wr_r >= 50), 
                            alpha=0.15, color="blue", interpolate=True)
            ax1.fill_between(timesteps, 50, wr_r, where=(wr_r < 50), 
                            alpha=0.15, color="red", interpolate=True)
        
        ax1.axhline(50, color="red", linestyle="--", alpha=0.7, linewidth=2)
        
        # Aggiungi bande sessioni
        for i, bound in enumerate(session_boundaries):
            ax1.axvspan(bound["start"], bound["end"], alpha=0.08, color=session_colors[i], zorder=0)
            ax1.axvline(bound["start"], color=session_colors[i], linestyle=":", alpha=0.5, linewidth=1)
        
        # Best e worst markers
        if data["wr_heuristic"] is not None:
            best_idx = np.argmax(data["wr_heuristic"])
            worst_idx = np.argmin(data["wr_heuristic"])
            ax1.scatter([timesteps[best_idx]], [data["wr_heuristic"][best_idx]*100], 
                       color="gold", s=150, marker="*", edgecolors="black", zorder=10, label=f"Best: {data['wr_heuristic'][best_idx]*100:.0f}%")
            
        ax1.set_ylabel("Win Rate %", fontsize=11, fontweight="bold")
        ax1.set_title("Win Rate Evolution", fontsize=12)
        ax1.set_ylim(0, 100)
        ax1.legend(loc="upper left", fontsize=9, ncol=3)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(timesteps[0], timesteps[-1])
        
        # Stats box
        if data["wr_random"] is not None:
            avg_wr_r = np.mean(data["wr_random"]) * 100
            max_wr_r = np.max(data["wr_random"]) * 100
            avg_wr_h = np.mean(data["wr_heuristic"]) * 100 if data["wr_heuristic"] is not None else 0
            max_wr_h = np.max(data["wr_heuristic"]) * 100 if data["wr_heuristic"] is not None else 0
            stats_text = f"Random: avg={avg_wr_r:.0f}% max={max_wr_r:.0f}%\nHeuristic: avg={avg_wr_h:.0f}% max={max_wr_h:.0f}%"
            ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # === SUBPLOT 2: Losses ===
        ax2 = axes3[1]
        
        if data["policy_loss"] is not None and data["value_loss"] is not None:
            ax2_twin = ax2.twinx()
            
            l1 = ax2.plot(timesteps, data["policy_loss"], 'b-', linewidth=1.5, alpha=0.8, label="Policy Loss")
            l2 = ax2_twin.plot(timesteps, data["value_loss"], 'r-', linewidth=1.5, alpha=0.8, label="Value Loss")
            
            ax2.set_ylabel("Policy Loss", color="blue", fontsize=10)
            ax2_twin.set_ylabel("Value Loss", color="red", fontsize=10)
            ax2.tick_params(axis='y', labelcolor="blue")
            ax2_twin.tick_params(axis='y', labelcolor="red")
            
            # Bande sessioni
            for i, bound in enumerate(session_boundaries):
                ax2.axvspan(bound["start"], bound["end"], alpha=0.08, color=session_colors[i], zorder=0)
            
            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc="upper right", fontsize=9)
        
        ax2.set_title("Training Losses", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(timesteps[0], timesteps[-1])
        
        # === SUBPLOT 3: Entropy & Entropy Coef ===
        ax3 = axes3[2]
        
        if data["entropy"] is not None:
            ax3_twin = ax3.twinx()
            
            l1 = ax3.plot(timesteps, data["entropy"], 'purple', linewidth=2, label="Policy Entropy")
            if data["ent_coef"] is not None:
                l2 = ax3_twin.plot(timesteps, data["ent_coef"], 'orange', linewidth=2, linestyle="--", label="Ent Coef (decay)")
                ax3_twin.set_ylabel("Entropy Coef", color="orange", fontsize=10)
                ax3_twin.tick_params(axis='y', labelcolor="orange")
            else:
                l2 = []
            
            ax3.set_ylabel("Policy Entropy", color="purple", fontsize=10)
            ax3.tick_params(axis='y', labelcolor="purple")
            
            # Bande sessioni
            for i, bound in enumerate(session_boundaries):
                ax3.axvspan(bound["start"], bound["end"], alpha=0.08, color=session_colors[i], zorder=0)
            
            lines = l1 + (l2 if l2 else [])
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc="upper right", fontsize=9)
        
        ax3.set_title("Entropy (exploration)", fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(timesteps[0], timesteps[-1])
        
        # === SUBPLOT 4: Learning Rate & FPS ===
        ax4 = axes3[3]
        
        if data["lr"] is not None:
            ax4_twin = ax4.twinx()
            
            l1 = ax4.plot(timesteps, data["lr"], 'green', linewidth=2, label="Learning Rate")
            if data["fps"] is not None:
                l2 = ax4_twin.plot(timesteps, data["fps"], 'gray', linewidth=1.5, alpha=0.6, label="FPS")
                ax4_twin.set_ylabel("FPS", color="gray", fontsize=10)
                ax4_twin.tick_params(axis='y', labelcolor="gray")
            else:
                l2 = []
            
            ax4.set_ylabel("Learning Rate", color="green", fontsize=10)
            ax4.tick_params(axis='y', labelcolor="green")
            ax4.set_xlabel("Timesteps", fontsize=11, fontweight="bold")
            
            # Bande sessioni (con label)
            for i, bound in enumerate(session_boundaries):
                ax4.axvspan(bound["start"], bound["end"], alpha=0.08, color=session_colors[i], zorder=0)
                mid = (bound["start"] + bound["end"]) / 2
                ax4.annotate(f"S{i+1}: {bound['label']}", xy=(mid, ax4.get_ylim()[0]), 
                           fontsize=8, ha="center", va="top", rotation=45, alpha=0.7)
            
            lines = l1 + (l2 if l2 else [])
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc="upper right", fontsize=9)
        
        ax4.set_title("Hyperparameters & Performance", fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(timesteps[0], timesteps[-1])
        
        plt.tight_layout()
        fig3.savefig(GRAPHS_DIR / "session_evolution.png", dpi=150)
        plt.close(fig3)
        print("✅ Salvato: session_evolution.png")
        
        # Stampa info sessioni
        print(f"\n-- SESSIONI DI TRAINING --")
        for i, s in enumerate(sessions):
            steps_k = s["duration_steps"] / 1000
            wr_r = s["final_wr_random"] * 100 if s["final_wr_random"] else 0
            wr_h = s["final_wr_heuristic"] * 100 if s["final_wr_heuristic"] else 0
            best_h = s["best_wr_heuristic"] * 100 if s["best_wr_heuristic"] else 0
            print(f"   S{i+1} [{s['label']}]: {steps_k:.0f}K | WR Rand: {wr_r:.0f}% | WR Heur: {wr_h:.0f}% (best: {best_h:.0f}%)")
    
    # === Summary ===
    print(f"\n{'='*50}")
    print("📈 STATISTICHE TRAINING")
    print(f"{'='*50}")
    print(f"   Timesteps totali: {timesteps[-1]:,}")
    if data["wr_random"] is not None:
        print(f"   Win rate vs random: {data['wr_random'][-1]*100:.1f}%")
    if data["wr_heuristic"] is not None:
        print(f"   Win rate vs heuristic: {data['wr_heuristic'][-1]*100:.1f}%")
    if data["value_loss"] is not None:
        print(f"   Value loss finale: {data['value_loss'][-1]:.3f}")
    if data["entropy"] is not None:
        print(f"   Entropy finale: {data['entropy'][-1]:.3f}")
    if data["fps"] is not None:
        print(f"   FPS medio: {np.mean(data['fps']):.0f}")
    
    print(f"\n✅ Grafici salvati in: {GRAPHS_DIR}")


def plot_monitor_results(x: np.ndarray, y: np.ndarray, episode_lengths: np.ndarray) -> None:
    """Genera grafici per Monitor logs (formato legacy SB3)."""
    ensure_dirs()
    backup_old_graphs()
    
    print(f"📊 Trovati {len(x):,} episodi")
    print(f"   Timesteps totali: {x[-1]:,}")
    
    # Aggregazione
    x_bin, y_mean, y_std, y_p25, y_p75 = bin_data(x, y)
    _, len_mean, len_std, _, _ = bin_data(x, episode_lengths)
    
    # Win rate per bin
    def estimate_win_rate(rewards, threshold=0):
        return (np.sum(rewards > threshold) / len(rewards)) * 100 if len(rewards) > 0 else 0
    
    bin_edges = np.linspace(x[0], x[-1], N_BINS + 1)
    win_rates = []
    for i in range(N_BINS):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1]) if i < N_BINS - 1 else (x >= bin_edges[i])
        if np.sum(mask) > 0:
            win_rates.append(estimate_win_rate(y[mask]))
    win_rates = np.array(win_rates) if win_rates else np.array([50])
    
    # FIGURA 1: Reward Analysis
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle(f"📈 Analisi Reward ({len(y):,} episodi)", fontsize=12, fontweight="bold")
    
    # 1.1 Reward
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
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Win Rate %")
    ax.set_title("Win Rate nel Tempo")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 1.3 Distribuzione
    ax = axes1[2]
    ax.hist(y, bins=50, edgecolor="black", alpha=0.7, color="steelblue", density=True)
    ax.axvline(np.mean(y), color="red", linestyle="-", linewidth=2, label=f"Media: {np.mean(y):.2f}")
    ax.axvline(np.median(y), color="orange", linestyle="--", linewidth=2, label=f"Mediana: {np.median(y):.2f}")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Densità")
    ax.set_title("Distribuzione Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1.savefig(GRAPHS_DIR / "reward_analysis.png", dpi=150)
    plt.close(fig1)
    
    print(f"✅ Grafici salvati in: {GRAPHS_DIR}")
    print(f"\n📊 Reward medio: {np.mean(y):.3f}")
    print(f"   Win rate finale: {win_rates[-1]:.1f}%")


def plot_results() -> None:
    """Genera i grafici dall'addestramento (auto-detect formato)."""
    ensure_dirs()
    
    # Prova prima train_recurrent logs
    recurrent_data = load_recurrent_logs()
    if recurrent_data is not None and len(recurrent_data["timesteps"]) > 0:
        print("📂 Rilevato formato: train_recurrent.py")
        plot_recurrent_results(recurrent_data)
        return
    
    # Fallback a Monitor logs
    monitor_data = load_monitor_logs()
    if monitor_data is not None:
        print("📂 Rilevato formato: SB3 Monitor (legacy)")
        x, y, lengths = monitor_data
        plot_monitor_results(x, y, lengths)
        return
    
    print("❌ Nessun log trovato!")
    print("   Percorsi cercati:")
    print(f"   - {LOGS_DIR / 'train_recurrent' / '*.csv'}")
    print(f"   - {LOGS_DIR / '*.monitor.csv'}")
    print("\n   Esegui prima un training con:")
    print("   python scripts/train_recurrent.py --timesteps 1000000")


def main():
    """Entry point CLI."""
    plot_results()


if __name__ == "__main__":
    main()
