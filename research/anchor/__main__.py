"""Experimental CLI: preparation is CPU-only; scoring remains separate."""
import argparse

from research.anchor.nested import run_nested


def main():
    parser = argparse.ArgumentParser(prog="python -m research.anchor")
    parser.add_argument("command", choices=("prepare", "run"))
    for key in ("registry", "groups", "config", "query", "output", "anchor-cache"):
        parser.add_argument("--"+key, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_nested(args, prepare_only=args.command == "prepare")


if __name__ == "__main__":
    main()
