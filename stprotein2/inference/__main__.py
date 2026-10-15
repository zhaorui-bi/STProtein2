"""Independent CLI so frozen training entry points remain unchanged."""
import argparse
import json

from stprotein2.inference.bundle import export_run, predict_file


def main(argv=None):
    parser = argparse.ArgumentParser(prog="stprotein2", description="Export a completed STProtein2 run or predict from raw RNA only")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Create a portable frozen model bundle")
    export.add_argument("--run", required=True); export.add_argument("--output", required=True)
    predict = commands.add_parser("predict", help="Predict one complete RNA slice without ADT")
    predict.add_argument("--bundle", required=True); predict.add_argument("--rna", required=True)
    predict.add_argument("--output", required=True); predict.add_argument("--sample", default="query")
    predict.add_argument("--device", default="cpu"); predict.add_argument("--threads", type=int, default=2)
    predict.add_argument("--execution", choices=("saved", "full", "adaptive"), default="saved")
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_run(args.run, args.output)
        print(json.dumps({"family": result["family"], "genes": len(result["genes"]), "proteins": len(result["proteins"])}))
    else:
        print(json.dumps(predict_file(args.bundle, args.rna, args.output, sample=args.sample,
                                     device=args.device, threads=args.threads, execution=args.execution)))


if __name__ == "__main__":
    main()
