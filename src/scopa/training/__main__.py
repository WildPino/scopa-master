"""
Entry point per training module.

Il training attivo è ora in scripts/train_recurrent.py.
Questo modulo è mantenuto per retrocompatibilità.

Usage:
    python scripts/train_recurrent.py --help
"""
import sys

def main():
    print("⚠️  Il training è ora gestito da scripts/train_recurrent.py")
    print()
    print("Uso:")
    print("  python scripts/train_recurrent.py --help        # Mostra opzioni")
    print("  python scripts/train_recurrent.py               # Training default")
    print("  python scripts/train_recurrent.py --fresh       # Training da zero")
    print("  python scripts/train_recurrent.py --benchmark   # Benchmark modello")
    sys.exit(0)

if __name__ == "__main__":
    main()
