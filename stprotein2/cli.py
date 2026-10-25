"""Train a reference baseline, export verified candidates and predict RNA slices."""
import argparse
import json
from pathlib import Path
from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog='stprotein2')
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('train', help='Fit B0 or a nested compact reference residual')
    fit.add_argument('--registry', required=True)
    fit.add_argument('--groups', required=True)
    fit.add_argument('--query', help='Exclude the complete source of this development query')
    fit.add_argument('--genes', type=int, default=4000)
    fit.add_argument('--output', required=True)
    fit.add_argument('--threads', type=int, default=2)
    fit.add_argument('--model', choices=('b0', 'own_mlp', 'neighbor_mlp'), default='b0')
    fit.add_argument('--config', help='Optional compact model/train JSON; CLI genes/model/threads apply')
    fit.add_argument('--device', default='cpu', help='Compact training device, e.g. cpu or cuda')
    export = commands.add_parser('export', help='Export a clean training run or independently verified research residual')
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
        if args.model == 'b0':
            if args.config or args.device != 'cpu':
                parser.error('--config and --device require a compact residual model')
            result = train(args.registry, args.groups, args.output, query=args.query, n_genes=args.genes, threads=args.threads)
        else:
            from .neural import ResidualConfig
            from .training import TrainConfig
            from .selection import train as train_residual
            spec = json.loads(Path(args.config).read_text()) if args.config else {}
            if set(spec) - {'model', 'train'}:
                raise ValueError('Compact configuration accepts only model and train fields')
            model_values = dict(spec.get('model', {}))
            for key, value in [('n_genes', args.genes), ('residual_head', args.model)]:
                if key in model_values and model_values[key] != value:
                    raise ValueError('Configuration differs from CLI ' + key)
                model_values[key] = value
            train_values = {**spec.get('train', {}), 'threads': args.threads}
            result = train_residual(args.registry, args.groups, args.output, query=args.query,
                                   model_config=ResidualConfig(**model_values), config=TrainConfig(**train_values), device=args.device)
    elif args.command == 'export':
        from .deployment import export_research, export_clean
        result = (export_research(args.run, args.verification, args.output) if args.verification
                  else export_clean(args.run, args.output))
    else:
        result = predict_file(args.bundle, args.rna, args.output, sample=args.sample, threads=args.threads)
    print(result)
