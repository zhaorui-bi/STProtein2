"""Repository evaluation commands, separate from the installable core CLI."""
import argparse
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='python -m evaluation', description='Evaluate completed protein predictions')
    commands = parser.add_subparsers(dest='command', metavar='COMMAND')
    commands.add_parser('run', help='Score a completed training run', add_help=False)
    commands.add_parser('prediction', help='Score standalone RNA-only prediction output', add_help=False)
    if not argv or argv[0] in {'-h', '--help'}:
        parser.print_help()
        return
    # Keep the previous flag-only training-run interface available.
    if argv[0] == 'run' or argv[0].startswith('--'):
        from .score import main as score
        return score(argv[1:] if argv[0] == 'run' else argv)
    if argv[0] == 'prediction':
        from .prediction import main as score
        return score(argv[1:])
    parser.error(f'unknown command: {argv[0]}')
