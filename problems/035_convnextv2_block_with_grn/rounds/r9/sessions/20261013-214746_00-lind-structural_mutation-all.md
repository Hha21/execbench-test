# Design session 20261013-214746_00-lind-structural_mutation-all

Task: {"round": "r9"}


### compile_b200 (draft 1): placeholder to check the call format only; kernel.cu is a stub


### run_tests (draft 2): r9 E1 v1: tc phase A = W1 only (preloaded, LN vectors + dwb/b2 via bulk copy), W2 conversion deferred into GEMM1 slack; sc: dwb/b2 via bulk copy


### probe_b200 (1): stamped probe of r9 v1 tc_kernel with runtime ablations (abl bit0 skip deferred W2, bit1 skip W1 conversion, bit2 skip prefill)


### run_tests (draft 3): r9 v2: W1 loads first (volatile), conv in their shadow, W1 convert last; W2 L2 prefetch in the sync-1 window; sc same reorder


### probe_b200 (2): patch persisted stamped probe to v2 (W1 loads first, conv in shadow, prefetch of W2); abl 0 full, 8 no prefetch, 1 no deferred W2, 2 no W1


### probe_b200 (3): stamped v2 tc probe: abl 0 full v2, 8 no prefetch, 32 no W1 stores, 2 no W1, 3 no W1 and no deferred W2


### run_tests (draft 4): r9 v3: v2 minus L2 prefetch, plus flat self-resetting atomic-counter barrier (module-level __device__ buffer) for both grid syncs in tc_kernel and sc_kernel


### run_tests (draft 5): r9 v4: tc_kernel with flat counter barrier + v2 phase A; sc_kernel back on cg grid.sync with the v2 weight-after-conv reorder
