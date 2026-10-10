# Design session 20261010-011624_02-structural_mutation-all

Task: {"round": "r2"}


### probe_b200 (1): 


### probe_b200 (2): 


### probe_b200 (3): 


### probe_b200 (4): 


### probe_b200 (5): 


### probe_b200 (6): 


### probe_b200 (7): 


### probe_b200 (8): 


### probe_b200 (9): 


### probe_b200 (10): 


### probe_b200 (11): 


### probe_b200 (12): 


### probe_b200 (13): 


### probe_b200 (14): 


### probe_b200 (15): 


### probe_b200 (16): 


### run_tests (draft 1): E3 R-init v2: u32 loads + shuffles + L2 bulk prefetch; dispatch 224 (>=6144) / 192 (>=3072) / 96 (>=1800) / lt


### run_tests (draft 2): E3 R-init v3: tile-0 post-add (no prefill exposure); 8192 group variants: 16x512/32x256/7976 = A (batch128+prefetch), 8x1024 = B (batch64), 64x128 = C (no prefetch)


### run_tests (draft 3): E3 R-init v4: I-cache test. 8192 group: 16x512/32x256/7976 = F (no-unroll batch64), 8x1024 = G (no-unroll batch128), 64x128 = A (unrolled control); 4096 group = 192 no-unroll batch64; 2048 = 96 no-unroll batch32


### run_tests (draft 4): E3 final candidate: L-band R-init t224 (rolled fill, batch128, prefetch) with 16x512 on the parent fused kernel as portal A/B control; S/M cublasLt as parent
