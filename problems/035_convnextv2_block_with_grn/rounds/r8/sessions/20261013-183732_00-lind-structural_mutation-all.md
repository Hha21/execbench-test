# Design session 20261013-183732_00-lind-structural_mutation-all

Task: {"round": "r8"}


### probe_b200 (1): r8 D E1: tc_kernel all-warp epilogue vs r6 epilogue, stamps, interleaved timing, precision


### probe_b200 (2): r8 D E1: epilogue ablations (no GELU / no stores / no pass2 / 2-deep tld)


### probe_b200 (3): r8 D E1: g built in smem during GEMM2 (no g tensor), all-warp GEMM1 epilogue pass 1 only


### probe_b200 (4): r8 D E1: decoupled g producers (warps 12-15) with TMA-landed gw*gf slices


### run_tests (draft 1): r8 D E1: all-warp GEMM1 epilogue + smem g (stage-owned producers) in tc_kernel; A&S GELU in sc_kernel; 4,96 added to sc table


### run_tests (draft 2): r8 D E1 final: all-warp GEMM1 epilogue with register-held h (g in global) in tc_kernel; A&S GELU in sc_kernel; 4,96 in sc table


### probe_b200 (5): r8 D E1: precision log for sc_kernel paths (A&S GELU, new 4,96 row)
