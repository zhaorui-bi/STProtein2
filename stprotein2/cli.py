"""Supported STProtein2 commands, with optional code loaded on dispatch."""
import argparse
import sys

from . import __version__


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='stprotein2', description='Spatial RNA-to-protein prediction')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    commands = parser.add_subparsers(dest='command', metavar='COMMAND')
    for name, help_text in (
        ('prepare', 'Prepare reference-only anchor caches'),
        ('train', 'Train and select the current STProtein2 model'),
        ('export', 'Export a completed model into a portable bundle'),
        ('predict', 'Predict proteins from a complete RNA slice'),
    ):
        commands.add_parser(name, help=help_text, add_help=False)
    if not argv or argv[0] in {'-h', '--help'}:
        parser.print_help()
        return
    if argv[0] == '--version':
        parser.parse_args(argv)
        return
    command, *arguments = argv
    if command in {'train', 'prepare'}:
        from .training.__main__ import main as train
        return train(['run' if command == 'train' else 'prepare', *arguments])
    if command in {'export', 'predict'}:
        from .inference.__main__ import main as inference
        return inference([command, *arguments])
    parser.error(f'unknown command: {command}')
