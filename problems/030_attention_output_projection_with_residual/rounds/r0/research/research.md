# Design session research

Task: {"round": "r0"}


### probe_b200 (1): 


### probe_b200 (2): 


### probe_b200 (3): 


### compile_b200 (draft 1): cublasLt fused residual first kernel: compile check incl. ld_flags


### run_tests (draft 2): first kernel: cublasLt fused residual (C=residual, D=output)


### probe_b200 (4): 


### run_tests (draft 3): first kernel: cublasLt fused residual, two-file layout


### compile_b200 (draft 5): CUTLASS sm100 feasibility compile (lint workaround)


### probe_b200 (5): 


### compile_b200 (draft 6): CUTLASS sm100 feasibility compile (manual strides)


### run_tests (draft 7): CUTLASS 1SM 128x256x64 persistent, fused residual: correctness + timing vs cublasLt


### compile_b200 (draft 8): CUTLASS 2SM 256x256 cluster 2x1 compile check


### run_tests (draft 9): CUTLASS 2SM 256x256 cluster 2x1 persistent, fused residual


### probe_b200 (6): 
