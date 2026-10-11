# Design session 20261011-004033_02-structural_mutation-m

Task: {"round": "r5"}


### probe_b200 (1): E3 k48 build + correctness + cycle timing vs k32


### probe_b200 (2): E3 diagnose Bpad384 nb1 k48: hot W, L2 promotion, hint, no-MMA


### run_tests (draft 1): r5-e3: k48 pair for 256<B (nb1 M, nb2 L), no CUTLASS


### probe_b200 (3): E3 follow-up: decoupled W (DRAM) / emb (L2) rings for k48


### probe_b200 (4): check /tmp persistence


### probe_b200 (5): E3: L2 prefetch of W ahead of the 3-stage k48 ring


### probe_b200 (6): E3: evict_first-hinted L2 prefetch of W


### compile_b200 (draft 2): static check of tc_k48.cu (identical to the tested file minus comments)
