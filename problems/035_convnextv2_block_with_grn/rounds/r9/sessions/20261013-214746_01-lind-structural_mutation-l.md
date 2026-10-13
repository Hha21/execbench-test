# Design session 20261013-214746_01-lind-structural_mutation-l

Task: {"round": "r9"}


### probe_b200 (1): r9 E2: cooperative+cluster launch, occupancy of 2-clusters, cta_group mixing compile


### compile_b200 (draft 1): 


### probe_b200 (2): check cta_group::2 TMA syntax compiles


### probe_b200 (3): check cta_group::2 TMA syntax compiles


### compile_b200 (draft 2): 


### run_tests (draft 4): r9 E2 v1: C=768 all-pair (GEMM1+GEMM2 cta_group::2), smem g producers


### probe_b200 (4): check probe file persistence


### probe_b200 (5): harness_time API


### probe_b200 (6): test whether a written file persists to the next probe


### probe_b200 (7): persistence check


### probe_b200 (8): write tc body/host files


### probe_b200 (9): dry run


### run_tests (draft 5): r9 E2 v2: cgf staged in smem per unit (one load round)


### probe_b200 (10): stamped pair kernel at C=768


### probe_b200 (11): noop


### run_tests (draft 6): r9 E2 v3: warp-per-stage g producers (4 stages in production), ready count 2


### run_tests (draft 7): r9 E2 v4: pair path only for C=768 B>=16; 2/8,768 back on the r8 1SM path (template PAIR)
