"""
Train - Script di addestramento per Scopa AI con logging su CSV

Modalità di training:
    - random: Avversario casuale (default)
    - self: Self-play (vs se stesso)
    - heuristic: Avversario euristico
    - mixed: Alterna tra le modalità

Compatibile con ARM64 (niente TensorBoard/gRPC).
"""
import os
import argparse
from datetime import datetime 
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from scopa_env import ScopaEnv

# Directory per i log
LOG_DIR = "./logs/"
MODEL_DIR = "./models/"


class SelfPlayCallback(BaseCallback):
    """
    Callback per aggiornare il modello usato per self-play.
    Aggiorna il riferimento al modello nell'ambiente ogni N step.
    """
    def __init__(self, update_freq=200000, verbose=0):
        super().__init__(verbose)
        self.update_freq = update_freq
    
    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            # Aggiorna il modello nell'ambiente
            env = self.training_env.envs[0]
            if hasattr(env, 'env'):  # Se wrapped (es. Monitor)
                env = env.env
            if hasattr(env, 'set_model'):
                env.set_model(self.model)
                if self.verbose > 0:
                    print(f"[SelfPlay] Modello aggiornato a step {self.n_calls}")
        return True

#DEPRECATED
def linear_schedule(initial_value):
    """
    Ritorna una funzione che riduce linearmente il valore 
    dal valore iniziale a 0.005 man mano che l'addestramento procede.
    """
    def func(progress_remaining):
        # progress_remaining va da 1.0 (inizio) a 0.0 (fine)
        return 0.005 + (initial_value - 0.005) * progress_remaining
    return func

def train(total_timesteps=1000000, opponent_mode='random', continue_from=None, use_gpu=False):
    """
    Addestra il modello MaskablePPO per la Scopa.
    
    Args:
        total_timesteps: Numero totale di step (18 step ≈ 1 partita)
        opponent_mode: Modalità avversario ('random', 'self', 'heuristic', 'mixed')
        continue_from: Path del modello da cui continuare (opzionale)
        use_gpu: Se True, usa CUDA per l'addestramento
    """
    device = "cuda" if use_gpu else "cpu"
    print(f"🖥️  Device: {device.upper()}")
    # Crea le directory se non esistono
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Crea l'ambiente con la modalità specificata
    env = ScopaEnv(opponent_mode=opponent_mode)
    env = Monitor(env, LOG_DIR, info_keywords=())

    #ent_schedule = linear_schedule(0.05)
    
    # 2. Definisci o carica il modello
    if continue_from:
        print(f"Caricamento modello da: {continue_from}")
        model = MaskablePPO.load(continue_from, env=env, ent_coef=0.07, learning_rate=0.0003, device=device)
    else:
        model = MaskablePPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=0.0003, #Per il training aritmetico 0.0001 | Normale 0.0003
            gamma=0.99,
            n_steps=2052,
            batch_size=256,
            n_epochs=10,
            ent_coef=0.05,
            device=device,
        )
    
    # 3. Setup callback per self-play
    callbacks = []
    if opponent_mode in ['self', 'mixed']:
        # Passa il modello all'ambiente per self-play
        base_env = env.env if hasattr(env, 'env') else env
        base_env.set_model(model)
        callbacks.append(SelfPlayCallback(update_freq=200000, verbose=1))
    
    # 4. Addestramento
    mode_emoji = {'random': '🎲', 'self': '🪞', 'heuristic': '🧠', 'mixed': '🔀'}
    print(f"\n{mode_emoji.get(opponent_mode, '❓')} Modalità: {opponent_mode.upper()}")
    print(f"🚀 Inizio addestramento per {total_timesteps} timestep...")
    print(f"   📊 Log salvati in: {LOG_DIR}")
    print(f"   💾 Modello salvato in: {MODEL_DIR}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks if callbacks else None
    )
    
    # 5. Salva il modello con timestamp e modalità
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"scopa_ai_{opponent_mode}_{timestamp}")
    model.save(model_path)
    
    # Salva anche una copia come "latest"
    model.save(os.path.join(MODEL_DIR, f"scopa_ai_{opponent_mode}_latest"))
    model.save(os.path.join(MODEL_DIR, "scopa_ai_latest"))
    
    print(f"\n✅ Addestramento completato!")
    print(f"   💾 Modello salvato: {model_path}.zip")
    print(f"   📊 Esegui 'python plot_training.py' per vedere i grafici")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Addestra Scopa AI')
    parser.add_argument('--mode', '-m', type=str, default='random',
                        choices=['random', 'self', 'heuristic', 'mixed'],
                        help='Modalità avversario (default: random)')
    parser.add_argument('--timesteps', '-t', type=int, default=1000000,
                        help='Numero totale di timesteps (default: 1000000)')
    parser.add_argument('--continue-from', '-c', type=str, 
                        default='./models/scopa_ai_latest',
                        help='Path del modello da cui continuare (default: ./models/scopa_ai_latest)')
    parser.add_argument('--fresh', '-f', action='store_true',
                        help='Ignora modello esistente e inizia da zero')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Usa GPU (CUDA) per l\'addestramento')
    
    args = parser.parse_args()
    
    # Gestisci il caso in cui il modello non esiste
    continue_from = args.continue_from
    if args.fresh:
        continue_from = None
        print("🆕 Avvio fresh (ignoro modelli esistenti)")
    elif continue_from and not os.path.exists(continue_from + '.zip'):
        print(f"⚠️  Modello non trovato: {continue_from}.zip")
        print("   Avvio nuovo addestramento da zero...")
        continue_from = None
    
    train(
        total_timesteps=args.timesteps,
        opponent_mode=args.mode,
        continue_from=continue_from,
        use_gpu=args.gpu
    )