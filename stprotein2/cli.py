"""Train a reference baseline, export verified candidates and predict RNA slices."""
import argparse
from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog='stprotein2')
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('train', help='Fit reference-only Ridge and the fixed spatial basis')
    fit.add_argument('--registry', required=True)
    fit.add_argument('--groups', required=True)
    fit.add_argument('--query', help='Exclude the complete source of this development query')
    fit.add_argument('--genes', type=int, default=4000)
    fit.add_argument('--output', required=True)
    fit.add_argument('--threads', type=int, default=2)
    export = commands.add_parser('export', help='Export a clean B0 run or an independently verified residual run')
    export.add_argument('--run', required=True)
    export.add_argument('--verification', help='Independent verification directory for a research residual run')
    export.add_argument('--output', required=True)
    predict = commands.add_parser('predict', help='Predict a whole RNA-only slice')
    predict.add_argument('--bundle', required=True)
    predict.add_argument('--rna', required=True)
    predict.add_argument('--sample', default='query')
    predict.add_argument('--output', required=True)
    predict.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    from .pipeline import train, predict_file
    if args.command == 'train':
        result = train(args.registry, args.groups, args.output, query=args.query, n_genes=args.genes, threads=args.threads)
    elif args.command == 'export':
        from .deployment import export_research, export_baseline
        result = (export_research(args.run, args.verification, args.output) if args.verification
                  else export_baseline(args.run, args.output))
    else:
        result = predict_file(args.bundle, args.rna, args.output, sample=args.sample, threads=args.threads)
    print(result)
