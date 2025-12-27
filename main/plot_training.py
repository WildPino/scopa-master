"""
Plot Training - Visualizza i progressi dell'addestramento

Legge i file CSV generati dal Monitor e crea grafici con matplotlib.
Mantiene una memoria storica in JSON per confrontare sessioni diverse.
Ottimizzato per grandi dataset (niente punti raw, solo aggregazioni).
"""
import os
import json
import shutil
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

LOG_DIR = "./logs/"
GRAPH_DIR = "./graphs/" 
HISTORY_FILE = "./logs/training_history.json"

# Numero di bin per aggregare i dati (regolabile)
N_BINS = 100

def moving_average(values, window_size):
    """Calcola la media mobile per lisciare il grafico."""
    if window_size <= 0 or len(values) < window_size:
        return values
    weights = np.ones(window_size) / window_size
    return np.convolve(values, weights, mode='valid')

def bin_data(x, y, n_bins=N_BINS):
    """Aggrega i dati in bin per una visualizzazione pulita."""
    if len(x) < n_bins:
        return x, y, np.zeros_like(y), np.zeros_like(y)
    
    bin_edges = np.linspace(x[0], x[-1], n_bins + 1)
    bin_means = []
    bin_stds = []
    bin_p25 = []
    bin_p75 = []
    bin_centers = []
    
    for i in range(n_bins):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        if i == n_bins - 1:  # Include ultimo punto
            mask = (x >= bin_edges[i]) & (x <= bin_edges[i + 1])
        
        if np.sum(mask) > 0:
            bin_values = y[mask]
            bin_means.append(np.mean(bin_values))
            bin_stds.append(np.std(bin_values))
            bin_p25.append(np.percentile(bin_values, 25))
            bin_p75.append(np.percentile(bin_values, 75))
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
    
    return np.array(bin_centers), np.array(bin_means), np.array(bin_stds), np.array(bin_p25), np.array(bin_p75)

def load_training_history():
    """Carica lo storico delle sessioni di training."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"sessions": []}
    return {"sessions": []}

def save_training_history(history):
    """Salva lo storico delle sessioni di training."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def backup_old_graphs():
    """Rinomina i grafici esistenti con suffisso _previous."""
    if not os.path.exists(GRAPH_DIR):
        return
    
    for filename in os.listdir(GRAPH_DIR):
        if filename.endswith('.png') and '_previous' not in filename:
            old_path = os.path.join(GRAPH_DIR, filename)
            previous_path = os.path.join(GRAPH_DIR, filename.replace('.png', '_previous.png'))
            if os.path.exists(previous_path):
                os.remove(previous_path)
            shutil.move(old_path, previous_path)
            print(f"📁 Backup: {filename} → {filename.replace('.png', '_previous.png')}")

def estimate_win_rate(rewards, threshold=0):
    """Stima approssimativa del win rate."""
    if len(rewards) == 0:
        return 0
    wins = sum(1 for r in rewards if r > threshold)
    return (wins / len(rewards)) * 100

def plot_results():
    """Genera i grafici dall'addestramento."""
    os.makedirs(GRAPH_DIR, exist_ok=True)
    backup_old_graphs()
    
    try:
        results = load_results(LOG_DIR)
        x, y = ts2xy(results, 'timesteps')
        x = np.array(x)
        y = np.array(y)
        episode_lengths = np.array(results['l'].values) if 'l' in results else np.array([])
    except Exception as e:
        print(f"❌ Errore nel caricamento dei log: {e}")
        return
    
    if len(x) == 0:
        print("❌ Nessun dato trovato nei log.")
        return
    
    print(f"📊 Trovati {len(x)} episodi completati")
    print(f"   Aggregazione in {N_BINS} bin per grafici leggibili")
    
    # === AGGIORNA MEMORIA STORICA ===
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
    print(f"💾 Sessione salvata nello storico ({len(history['sessions'])} sessioni totali)")
    
    # === AGGREGAZIONE DATI ===
    x_bin, y_mean, y_std, y_p25, y_p75 = bin_data(x, y, N_BINS)
    
    if len(episode_lengths) > 0:
        _, len_mean, len_std, _, _ = bin_data(x, episode_lengths, N_BINS)
    
    # Calcola win rate per bin
    win_rates = []
    bin_edges = np.linspace(x[0], x[-1], N_BINS + 1)
    for i in range(N_BINS):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        if i == N_BINS - 1:
            mask = (x >= bin_edges[i]) & (x <= bin_edges[i + 1])
        if np.sum(mask) > 0:
            win_rates.append(estimate_win_rate(y[mask]))
    win_rates = np.array(win_rates)
    
    # === FIGURA 1: REWARD ANALYSIS (3 grafici) ===
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle(f'📈 Analisi Reward ({len(y):,} episodi → {N_BINS} bin)', fontsize=12, fontweight='bold')
    
    # 1.1 Reward con banda di confidenza
    ax = axes1[0]
    ax.fill_between(x_bin, y_p25, y_p75, alpha=0.3, color='blue', label='25°-75° percentile')
    ax.plot(x_bin, y_mean, color='darkblue', linewidth=2, label='Media')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Reward')
    ax.set_title('Reward nel Tempo')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 1.2 Win Rate
    ax = axes1[1]
    ax.plot(x_bin, win_rates, color='green', linewidth=2)
    ax.axhline(50, color='red', linestyle='--', alpha=0.7, label='50%')
    ax.fill_between(x_bin, 50, win_rates, where=(win_rates >= 50), 
                    alpha=0.3, color='green', interpolate=True)
    ax.fill_between(x_bin, 50, win_rates, where=(win_rates < 50), 
                    alpha=0.3, color='red', interpolate=True)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Win Rate %')
    ax.set_title('Win Rate nel Tempo')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 1.3 Distribuzione Reward
    ax = axes1[2]
    ax.hist(y, bins=50, edgecolor='black', alpha=0.7, color='steelblue', density=True)
    ax.axvline(np.mean(y), color='red', linestyle='-', linewidth=2, label=f'Media: {np.mean(y):.2f}')
    ax.axvline(np.median(y), color='orange', linestyle='--', linewidth=2, label=f'Mediana: {np.median(y):.2f}')
    ax.axvline(0, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Reward')
    ax.set_ylabel('Densità')
    ax.set_title('Distribuzione Reward')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1.savefig(os.path.join(GRAPH_DIR, 'reward_analysis.png'), dpi=150)
    plt.close(fig1)
    
    # === FIGURA 2: PERFORMANCE (3 grafici) ===
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('📊 Analisi Performance', fontsize=12, fontweight='bold')
    
    # 2.1 Episode Length
    ax = axes2[0]
    if len(episode_lengths) > 0:
        ax.plot(x_bin, len_mean, color='teal', linewidth=2, label='Media')
        ax.fill_between(x_bin, len_mean - len_std, len_mean + len_std, 
                        alpha=0.3, color='teal', label='±1 std')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Lunghezza Episodio')
        ax.set_title('Lunghezza Episodi')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Dati non disponibili', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)
    
    # 2.2 Cumulative Reward
    ax = axes2[1]
    cumulative = np.cumsum(y)
    # Downsample per visualizzazione
    step = max(1, len(cumulative) // 500)
    ax.plot(x[::step], cumulative[::step], color='darkblue', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(x[::step], 0, cumulative[::step], 
                    where=(cumulative[::step] >= 0), alpha=0.2, color='green')
    ax.fill_between(x[::step], 0, cumulative[::step], 
                    where=(cumulative[::step] < 0), alpha=0.2, color='red')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Reward Cumulativo')
    ax.set_title(f'Reward Cumulativo: {cumulative[-1]:.0f}')
    ax.grid(True, alpha=0.3)
    
    # 2.3 Heatmap Reward vs Episode Length
    ax = axes2[2]
    if len(episode_lengths) > 0:
        h, xedges, yedges = np.histogram2d(episode_lengths, y, bins=30)
        im = ax.imshow(h.T, origin='lower', aspect='auto', cmap='viridis',
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.colorbar(im, ax=ax, label='Frequenza')
        ax.set_xlabel('Lunghezza Episodio')
        ax.set_ylabel('Reward')
        ax.set_title('Heatmap Length vs Reward')
    else:
        ax.text(0.5, 0.5, 'Dati non disponibili', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    fig2.savefig(os.path.join(GRAPH_DIR, 'performance_analysis.png'), dpi=150)
    plt.close(fig2)
    
    # === FIGURA 3: STORICO SESSIONI ===
    if len(history["sessions"]) > 1:
        sessions = history["sessions"]
        n_sessions = len(sessions)
        
        fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
        fig3.suptitle(f'📅 Storico: {n_sessions} Sessioni di Training', fontsize=12, fontweight='bold')
        
        session_nums = list(range(1, n_sessions + 1))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_sessions))
        
        # 3.1 Reward medio
        ax = axes3[0, 0]
        means = [s["mean_reward"] for s in sessions]
        bars = ax.bar(session_nums, means, color=colors, edgecolor='black', alpha=0.8)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(np.mean(means), color='red', linestyle='-', alpha=0.7, label=f'Media globale: {np.mean(means):.2f}')
        ax.set_xlabel('Sessione')
        ax.set_ylabel('Reward Medio')
        ax.set_title('Reward Medio per Sessione')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3.2 Win Rate
        ax = axes3[0, 1]
        win_rates_hist = [s.get("win_rate_approx", 0) for s in sessions]
        ax.bar(session_nums, win_rates_hist, color=colors, edgecolor='black', alpha=0.8)
        ax.axhline(50, color='red', linestyle='--', alpha=0.7, label='50%')
        ax.set_xlabel('Sessione')
        ax.set_ylabel('Win Rate %')
        ax.set_title('Win Rate per Sessione')
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3.3 Trend complessivo
        ax = axes3[1, 0]
        cumulative_ts = np.cumsum([s.get("total_timesteps", 0) for s in sessions])
        ax.plot(session_nums, means, 'o-', color='steelblue', linewidth=2, markersize=8, label='Reward')
        
        # Trendline
        if n_sessions >= 3:
            z = np.polyfit(session_nums, means, 1)
            p = np.poly1d(z)
            ax.plot(session_nums, p(session_nums), '--', color='red', alpha=0.7, 
                    label=f'Trend: {z[0]:+.3f}/sessione')
        
        ax.set_xlabel('Sessione')
        ax.set_ylabel('Reward Medio')
        ax.set_title('Evoluzione nel Tempo')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 3.4 Timesteps cumulativi
        ax = axes3[1, 1]
        ax.bar(session_nums, [s.get("total_timesteps", 0) for s in sessions], 
               color=colors, edgecolor='black', alpha=0.6, label='Per sessione')
        ax.plot(session_nums, cumulative_ts, 'ro-', linewidth=2, markersize=6, label='Cumulativo')
        ax.set_xlabel('Sessione')
        ax.set_ylabel('Timesteps')
        ax.set_title(f'Timesteps Totali: {cumulative_ts[-1]:,}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig3.savefig(os.path.join(GRAPH_DIR, 'session_history.png'), dpi=150)
        plt.close(fig3)
        print(f"✅ Grafico storico sessioni salvato!")
    
    print(f"\n✅ Grafici salvati in: {GRAPH_DIR}")
    
    # === STATISTICHE ===
    print(f"\n{'='*50}")
    print(f"📈 STATISTICHE SESSIONE CORRENTE")
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

if __name__ == "__main__":
    plot_results()
