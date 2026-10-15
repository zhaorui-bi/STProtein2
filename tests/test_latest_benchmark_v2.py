"""Recovery preserves cohort rules and accepts an actual unindexed CUDA device."""
import argparse
import pytest
import torch

import test_latest_benchmark as original
import stprotein2.training.nested as nested
import stprotein2.training.__main__ as cli


def test_v2_four_source_training_and_CF0_feasibility(tmp_path,monkeypatch):
    monkeypatch.setattr(original,'nested',nested)
    original.test_four_source_actual_training_closed_gates_and_missing_query_ADT(tmp_path,monkeypatch)
    original.test_CF0_still_requires_six_groups_before_accessing_registry(tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_actual_cuda_allocator_accepts_unindexed_device(monkeypatch):
    import sys
    arguments=['latest','run','--registry','unused','--groups','unused','--config','unused',
        '--query','unused','--output','unused','--anchor-cache','unused','--device','cuda',
        '--cuda-allocator-gib','4']
    monkeypatch.setattr(sys,'argv',arguments)
    calls=[]
    def checked(args,prepare_only):
        assert args.device=='cuda' and args.cuda_allocator_gib==4.
        # Only a tiny actual allocation; the parser must have set its4GiB cap.
        value=torch.ones(16,device='cuda');assert float(value.sum())==16.
        calls.append(True)
    monkeypatch.setattr(cli,'run_nested',checked)
    cli.main();assert calls==[True]
