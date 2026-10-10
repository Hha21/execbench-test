# Design session research

Task: {"round": "r0"}


### probe_b200 (1): reference kernels + timing at B=5,128,919; TF32 for info


### probe_b200 (2): FP32 SIMT peak: FFMA vs packed FFMA2 (fma.rn.f32x2)


### probe_b200 (3): reference kernels; chunked x6 vs full cuBLAS; TF32 rate


### probe_b200 (4): cublas emulation enums + version; mma.sync tf32 rate on sm_100


### compile_b200 (draft 1): first candidate: cublasLt fp32 + bias epilogue, 6 calls


### probe_b200 (5): cublasLt via ctypes: fp32 full-N + bias epilogue; BF16x9 emulated speed and error; kernel names


### run_tests (draft 2): first candidate: cuBLASLt fp32 full-N GEMM + bias epilogue into workspace, then one split kernel


### probe_b200 (6): 3xTF32 via pre-split + 3 cuBLAS TF32 GEMMs: time and error vs fp64


### probe_b200 (7): custom smem-staged skinny SIMT kernel for B<=32 (memory-bound band)


### probe_b200 (8): skinny SIMT kernel for B<=32, retry with smem= kwarg


### probe_b200 (9): skinny SIMT kernel for B<=32, set max dynamic smem attribute


### run_tests (draft 3): dispatch: B<=8 skinny<8>, B<=16 skinny<16>, B<=128 cuBLASLt fp32+scatter, B>128 3xTF32 (split + 3 TF32 GEMMs + scatter)


### run_tests (draft 4): same kernel, 3xTF32 threshold lowered to B > 64 (96 and 128 move to 3xTF32)


### compile_b200 (draft 6): static features of the final dispatch kernel (both files)
