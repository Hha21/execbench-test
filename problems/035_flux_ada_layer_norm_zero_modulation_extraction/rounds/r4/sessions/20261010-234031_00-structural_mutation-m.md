# Design session 20261010-234031_00-structural_mutation-m

Task: {"round": "r4"}


### probe_b200 (1): env discovery: harness_time signature, CUTLASS headers, nvcc


### probe_b200 (2): E1: build tc_pair2 (two accumulators), correctness vs fp64, cycles sweep full/feed-only/resident


### probe_b200 (3): E1: build CUTLASS 256x192/256x256 tiles into /tmp (check build time and whether /tmp persists)


### probe_b200 (4): check /tmp persistence


### probe_b200 (5): E1: split/order diagnostics at kbs1, ramp/epilogue breakdown, interleaved harness_time vs CUTLASS tiles, NVML clock


### probe_b200 (6): E1: rebuild CUTLASS tiles in this container


### probe_b200 (7): container check


### probe_b200 (8): E1: split/order diagnostics, ramp/epilogue, interleaved harness_time vs CUTLASS, NVML clock


### probe_b200 (9): E1: floor ablation (spin wait, commit every 2nd stage, idle-warp sleep, epilogue x32)


### probe_b200 (10): E1: floor ablation (spin, commit pairs, idle sleep, epilogue x32), CUTLASS build in background


### probe_b200 (11): container check


### probe_b200 (12): E1: wait-pair/triple MMA loop, no-fence, vs CUTLASS (harness_time, nvidia-smi clock)


### run_tests (draft 1): r4-e1: tc_pair2 (two TMEM accumulators, one pass over N=Bpad) for 256<B<=512; tc_small/tc_pair unchanged; CUTLASS 256x192 for B>512
