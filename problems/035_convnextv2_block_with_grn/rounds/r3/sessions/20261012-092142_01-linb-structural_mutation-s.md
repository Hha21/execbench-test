# Design session 20261012-092142_01-linb-structural_mutation-s

Task: {"round": "r3"}


### probe_b200 (1): API discovery


### probe_b200 (2): Kernel launcher API


### probe_b200 (3): E2 step1 microbench: post-sync load round 18 cells


### probe_b200 (4): E2 step1 microbench fixed + barrier variants


### probe_b200 (5): E2 step1 microbench, fixed epoch


### probe_b200 (6): E2 step1 load-round cells


### probe_b200 (7): check build env persistence


### probe_b200 (8): noop


### probe_b200 (9): stash part 1 of kernel (testing persistence of /tmp)


### probe_b200 (10): stash part 2 (ring phase 1)


### probe_b200 (11): stash part 3 (ring GEMM, verbatim)


### probe_b200 (12): persistence check


### probe_b200 (13): E2 step2: flat-new (bulk rows + prefetch + conv-first) vs flat-old, interleaved


### probe_b200 (14): E2 ablation of new-flat features


### probe_b200 (15): placeholder


### probe_b200 (16): final kernel part 1 (check same-container write)


### run_tests (draft 1): r3 E2 candidate: bulk-row post-sync loads on flat path, dispatch 1,192 (both) + 1,160/4,96 (GEMM2)


### probe_b200 (17): noop guard


### probe_b200 (18): ring part (cannot persist; checking size only)


### probe_b200 (19): wrapper only


### run_tests (draft 2): control: same build, parent dispatch, no bulk


### run_tests (draft 3): final: parent dispatch + GEMM2 bulk rows on 1,160 / 4,96
