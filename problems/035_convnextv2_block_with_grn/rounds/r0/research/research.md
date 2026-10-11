# Design session research

Task: {"round": "r0"}


### probe_b200 (1): precision simulation: fp32 ref vs fp64, tf32, fp16x3 split against tolerance


### probe_b200 (2): FP32 SIMT peak: FFMA vs fma.rn.f32x2 throughput at current clock


### probe_b200 (3): reference kernel breakdown with torch.profiler at two sizes; check harness io generators


### probe_b200 (4): launch overhead: empty kernel, 2 launches, cooperative-style atomic grid barrier with 296 CTAs under harness timing


### probe_b200 (5): launch overhead rerun: empty kernel, 2-3 launches, atomic grid barrier with 296/1184 CTAs


### probe_b200 (6): inspect harness io.py generators and find the gen_inputs API


### probe_b200 (7): cuBLAS fp32 SIMT sgemm rates for both GEMMs of every workload (warm, CUDA events)


### probe_b200 (8): rest of io.py heuristic generation, _rand_tensor, and gen_inputs signature


### probe_b200 (9): precision simulation v2 with the harness's real input distributions: tf32/fp16/bf16 single and split variants per GEMM


### compile_b200 (draft 1): first draft: cooperative 3-phase fp32 SIMT kernel


### run_tests (draft 2): r0 research anchor: cooperative 3-phase fp32 SIMT kernel, first correctness+timing run


### probe_b200 (10): inspect b200probe launcher API (dynamic smem support, arg handling)


### probe_b200 (11): inspect b200probe Kernel class: launch signature, shared memory, cooperative support


### probe_b200 (12): per-phase breakdown of the r0 kernel: phases launched as separate kernels, harness timing, 4 workloads


### probe_b200 (13): per-phase breakdown of the r0 kernel (retry without stdint.h)


### compile_b200 (draft 3): r0 anchor v2: patch dwconv, K-major activations, 4 GEMM configs incl. warp split-K, cross-CTA split-K phase 3


### run_tests (draft 4): r0 anchor v2: patch dwconv + K-major activations + split-K configs


### probe_b200 (14): per-phase breakdown of v2 kernel (patch dwconv, K-major, split-K) as separate launches; chosen configs printed


### probe_b200 (15): GEMM core microbenchmark: 128x128x32 SIMT tile, K-major A; lane-layout, unroll, register-cap and no-global variants; TF/s at M=131072


### probe_b200 (16): phase-1 components for 32,768,7,7 and 64,96,112,112: transposes alone, mode-0 twr variants, patch mode; harness timing


### probe_b200 (17): GEMM phase variants: launch-bounds 1 vs 2, no-epilogue, no-prologue, no-affine; phase 2/3 times at three shapes


### probe_b200 (18): phase-1 improvements: warp-level transposes, zero-padded smem planes for small images, prefetching patch conv; correctness vs conv2d and harness timing
