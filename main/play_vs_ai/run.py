"""
Run script for Scopa vs AI game
"""
import os
import sys

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from game_gui import ScopaGUI


def main():
    print("=" * 50)
    print("  SCOPA vs AI - Gioca contro l'Intelligenza Artificiale")
    print("=" * 50)
    print()
    print("Controlli:")
    print("  - Clicca una carta per giocarla")
    print("  - Se ci sono più combinazioni, seleziona le carte da prendere")
    print("  - INVIO per confermare la presa")
    print("  - ESC per annullare/uscire")
    print("  - SPAZIO per nuova partita (a fine gioco)")
    print()
    print("Avvio gioco...")
    
    game = ScopaGUI()
    game.run()


if __name__ == "__main__":
    main()
