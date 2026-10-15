"""Compatibility wrapper; use stprotein2 score for the public interface."""
from stprotein2.evaluation.score import main, score

if __name__ == "__main__":
    main()
