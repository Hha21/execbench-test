# Design session 20261008-135844_03-structural_mutation-m

Task: {"round": "r13"}


### probe_b200 (1): inspect harness_time / clear cache


### probe_b200 (2): cuda_kernel api


### probe_b200 (3): Kernel api


### probe_b200 (4): E4: bulk-load one-shot variants, correctness + 586 tok dirty/clean


### probe_b200 (5): E4: 1024/2048/128 dirty vs clean


### probe_b200 (6): E4: T=2/T=4 bulk vs c2 at 1024-8192


### compile_b200 (draft 1): r13-e4 dispatch: c2 + bulk T=2 at 1100<tok<=2100


### run_tests (draft 2): r13-e4-bulkld-t2-disp full test


### run_tests (draft 3): r13-e4-bulkld-t2-disp rerun (CUPTI flake on S workloads)
