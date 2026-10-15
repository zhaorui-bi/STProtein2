"""Experimental CLI: preparation is CPU-only; scoring remains separate."""
import argparse

from research.strict_nested import run_nested


def main():
    parser = argparse.ArgumentParser(prog="python -m research.strict_cli")
    parser.add_argument("command", choices=("prepare", "run"))
    for key in ("registry", "groups", "config", "query", "output", "anchor-cache"):
        parser.add_argument("--"+key, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-anchor-cache", action="append", default=[],
                        help="Read-only independent anchor cache; repeat to search multiple caches")
    args = parser.parse_args()
    run_nested(args, prepare_only=args.command == "prepare")


if __name__ == "__main__":
    main()
