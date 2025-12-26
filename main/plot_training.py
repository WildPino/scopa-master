"""
Plot Training - Visualizza i progressi dell'addestramento

Legge i file CSV generati dal Monitor e crea grafici con matplotlib.
I grafici vecchi vengono rinominati con suffisso _previous.
"""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

LOG_DIR = "./logs/"
GRAPH_DIR = "./graphs/" 

def moving_average(values, window_size):
    """Calcola la media mobile per lisciare il grafico."""
    weights = np.ones(window_size) / window_size
    return np.convolve(values, weights, mode='valid')

def backup_old_graphs():
    """Rinomina i grafici esistenti con suffisso _previous."""
    if not os.path.exists(GRAPH_DIR):
        return
    
    for filename in os.listdir(GRAPH_DIR):
        if filename.endswith('.png') and '_previous' not in filename:
            old_path = os.path.join(GRAPH_DIR, filename)
            # Rimuovi eventuali _previous esistenti
            previous_path = os.path.join(GRAPH_DIR, filename.replace('.png', '_previous.png'))
            if os.path.exists(previous_path):
                os.remove(previous_path)
            # Rinomina il vecchio grafico
            shutil.move(old_path, previous_path)
            print(f"📁 Backup: {filename} → {filename.replace('.png', '_previous.png')}")

def plot_results():
    """Genera i grafici dall'addestramento."""
    os.makedirs(GRAPH_DIR, exist_ok=True)
    
    # Backup dei grafici precedenti
    backup_old_graphs()
    
    try:
        # Carica i risultati dal Monitor
        results = load_results(LOG_DIR)
        x, y = ts2xy(results, 'timesteps')
    except Exception as e:
        print(f"❌ Errore nel caricamento dei log: {e}")
        print(f"   Assicurati di aver eseguito almeno un addestramento.")
        return
    
    if len(x) == 0:
        print("❌ Nessun dato trovato nei log.")
        return
    
    print(f"📊 Trovati {len(x)} episodi completati")
    
    # --- GRAFICO 1: Reward per episodio ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, y, alpha=0.3, color='blue', label='Reward (raw)')
    
    # Media mobile per lisciare
    window = min(50, len(y) // 4) if len(y) > 4 else 1
    if window > 1:
        y_smooth = moving_average(y, window)
        x_smooth = x[window-1:]
        plt.plot(x_smooth, y_smooth, color='red', linewidth=2, label=f'Media mobile ({window})')
    
    plt.xlabel('Timesteps')
    plt.ylabel('Reward (punti a fine partita)')
    plt.title('📈 Reward per Episodio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # --- GRAFICO 2: Distribuzione dei reward ---
    plt.subplot(1, 2, 2)
    plt.hist(y, bins=20, edgecolor='black', alpha=0.7, color='green')
    plt.axvline(np.mean(y), color='red', linestyle='--', linewidth=2, label=f'Media: {np.mean(y):.2f}')
    plt.xlabel('Reward')
    plt.ylabel('Frequenza')
    plt.title('📊 Distribuzione dei Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salva il grafico
    graph_path = os.path.join(GRAPH_DIR, 'training_progress.png')
    plt.savefig(graph_path, dpi=150)
    plt.close()
    
    print(f"\n✅ Grafici salvati in: {graph_path}")
    
    # --- STATISTICHE ---
    print(f"\n📈 STATISTICHE ADDESTRAMENTO:")
    print(f"   Episodi totali: {len(y)}")
    print(f"   Reward medio: {np.mean(y):.2f}")
    print(f"   Reward max: {np.max(y):.2f}")
    print(f"   Reward min: {np.min(y):.2f}")
    print(f"   Deviazione std: {np.std(y):.2f}")
    
    # Trend (ultimi 50 vs primi 50)
    if len(y) >= 100:
        early = np.mean(y[:50])
        late = np.mean(y[-50:])
        improvement = late - early
        emoji = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
        print(f"   {emoji} Trend: {early:.2f} → {late:.2f} ({improvement:+.2f})")

if __name__ == "__main__":
    plot_results()
