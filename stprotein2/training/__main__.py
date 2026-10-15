"""Bounded allocator worker; do not silently switch device or training settings."""
import argparse
import math
import torch

from stprotein2.training.nested import run_nested


def main(argv=None):
    public = argv is not None
    prog = ('stprotein2 prepare' if argv and argv[0] == 'prepare' else 'stprotein2 train') if public else 'stprotein2-train'
    parser=argparse.ArgumentParser(prog=prog)
    parser.add_argument('command',choices=('prepare','run'), help=argparse.SUPPRESS if public else 'Training stage')
    for key in ('registry','groups','config','query','output','anchor-cache'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--reuse-anchor-cache',action='append',default=[])
    parser.add_argument('--cuda-allocator-gib',type=float,default=4.)
    args=parser.parse_args(argv)
    if not math.isfinite(args.cuda_allocator_gib) or args.cuda_allocator_gib<=0:
        parser.error('Allocator ceiling must be finite and positive')
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA unavailable')
        device=torch.device(args.device)
        total=torch.cuda.get_device_properties(device).total_memory
        device_index=torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.set_per_process_memory_fraction(min(1.,args.cuda_allocator_gib*1024**3/total),device_index)
    run_nested(args,prepare_only=args.command=='prepare')


if __name__=='__main__':
    main()
