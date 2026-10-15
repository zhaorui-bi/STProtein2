"""Compatibility script; prefer python -m evaluation from the repository root."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.score import main, score

if __name__ == "__main__":
    main()
