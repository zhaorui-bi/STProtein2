# Latest benchmark CUDA launch recovery

The first controller and child are terminal. The child failed in allocator setup
before reading data or fitting: PyTorch `set_per_process_memory_fraction` rejects
`torch.device('cuda')` without an index. No accuracy result was generated.
Preserve `runs/latest_v2_1_benchmark_v1`, its frozen package and failure logs.

Version2 resolves a missing index with `torch.cuda.current_device()` before the
allocator call. Student/model math, initialization, losses, split rules, seeds,
maximum epochs, gate thresholds, 4GiB allocator ceiling and admission are unchanged.
The version2 package and controller use a fresh output directory and source hashes.
CPU benchmark adapter tests are rerun against the version2 namespace; a CUDA
integration check verifies that actual unindexed `cuda` now accepts the allocator
setting. Observation timeout was not a restart criterion: the original process
and fresh non-OOM traceback explicitly establish termination.
