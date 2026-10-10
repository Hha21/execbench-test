### Rationale
The one-launch batched pointer-array variant is refuted, so the candidate is the two-launch path: one cuBLASLt `FAST_TF32` GEMM with bias epilogue into `tmp`, then the existing scatter kernel. It passes 16/16 on the rented B200. The emulator predicts a portal score of 0.537 ± 0.011, against 0.235 for r0.

The batched pointer-array variant lost at every B. The speed comparison was at the cuBLASLt algorithm's first heuristic choice and its best of the 7–8 returned, with pinned-memory pointer arrays and 3 interleaved rounds. At B = 919 it took 178 µs against 157 µs for the big GEMM plus scatter (+14%, past the 10% refutation line). At B = 5 it took 63 µs against 55 µs (+15%).

Two things I did not cleanly pin down:
- **Heuristic choice:** I used heuristic 0 everywhere. It was fastest or within noise at every B I timed (5, 16, 32, 128, 384, 919), so I did not build a per-B table.
- **Error margin:** B = 128 reaches 1.75e-3 max |err| against fp64, just over your 1.7e-3 success bar, though no element exceeds atol.

```json solution-spec
{"name": "r1-e1-lt-fasttf32-bias-scatter", "definition": "035_flux_ada_layer_norm_zero_modulation_extraction", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": [], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"], "ld_flags": ["-lcublasLt"]}}}
```

```cuda file=kernel.cu
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdint>
#include <map>
#include <mutex>

// AdaLayerNormZero modulation extraction: out_c[b, j] = sum_k emb[b,k] * W[c*3072+j, k] + bias[c*3072+j], c = 0..5.
// One path for every B: cuBLASLt CUBLAS_COMPUTE_32F_FAST_TF32 (fp32 in/out, fp32 accumulate) with the bias epilogue
// into tmp[B, 18432], then scatter6_kernel splits the columns into the six output tensors (2 launches).

namespace {
constexpr int kD = 3072;        // inner_dim
constexpr int kN = 6 * kD;      // output_dim = 18432

struct Plan {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulAlgo_t algo;
  bool have_algo = false;
};

cublasLtHandle_t g_handle = nullptr;
std::mutex g_mu;
std::map<int, Plan> g_plans;   // key = B

Plan& get_plan(int B, size_t ws_bytes) {
  if (!g_handle) cublasLtCreate(&g_handle);
  auto it = g_plans.find(B);
  if (it != g_plans.end()) return it->second;
  Plan p;
  cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F);
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
  // Column-major view: A = W as K x N (ld K), op T; B = emb as K x B (ld K); D = N x B (ld N) == row-major [B, N].
  cublasLtMatrixLayoutCreate(&p.la, CUDA_R_32F, kD, kN, kD);
  cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_32F, kD, B, kD);
  cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_32F, kN, B, kN);
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes));
  cublasLtMatmulHeuristicResult_t res[1];
  int n = 0;
  if (cublasLtMatmulAlgoGetHeuristic(g_handle, p.desc, p.la, p.lb, p.lc, p.lc, pref, 1, res, &n) == CUBLAS_STATUS_SUCCESS && n > 0) {
    p.algo = res[0].algo;
    p.have_algo = true;
  }
  cublasLtMatmulPreferenceDestroy(pref);
  return g_plans.emplace(B, p).first->second;
}

struct OutPtrs { float* p[6]; };

// tmp[B, 18432] -> six [B, 3072] outputs, 16 B per thread.
__global__ void __launch_bounds__(256) scatter6_kernel(const float4* __restrict__ src, OutPtrs outs, int B) {
  constexpr int kN4 = kN / 4;    // 4608 float4 per row
  constexpr int kD4 = kD / 4;    // 768 float4 per chunk row
  const size_t total = (size_t)B * kN4;
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int b = (int)(i / kN4);
  const int n4 = (int)(i - (size_t)b * kN4);
  const int c = n4 / kD4;
  const int j = n4 - c * kD4;
  const float4 v = __ldcs(src + i);
  reinterpret_cast<float4*>(outs.p[c])[(size_t)b * kD4 + j] = v;
}
}  // namespace

// Scratch (allocated by the caller through the PyTorch allocator; no kernels): tmp [B, 18432] and ws.
extern "C" int e1_launch(const float* emb, const float* weight, const float* bias, float* const* outs, int B,
                         float* tmp, void* ws, size_t ws_bytes, cudaStream_t stream) {
  OutPtrs op;
  for (int i = 0; i < 6; ++i) op.p[i] = outs[i];
  {
    std::lock_guard<std::mutex> lk(g_mu);
    Plan& p = get_plan(B, ws_bytes);
    const float alpha = 1.f, beta = 0.f;
    cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    cublasStatus_t st = cublasLtMatmul(g_handle, p.desc, &alpha, weight, p.la, emb, p.lb, &beta, tmp, p.lc, tmp, p.lc,
                                       p.have_algo ? &p.algo : nullptr, ws, ws_bytes, stream);
    if (st != CUBLAS_STATUS_SUCCESS) return 1000 + (int)st;
  }
  const size_t total = (size_t)B * (kN / 4);
  scatter6_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(reinterpret_cast<const float4*>(tmp), op, B);
  return (int)cudaGetLastError();
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

extern "C" int e1_launch(const float* emb, const float* weight, const float* bias, float* const* outs, int B,
                         float* tmp, void* ws, size_t ws_bytes, cudaStream_t stream);

void run(torch::Tensor emb, torch::Tensor weight, torch::Tensor bias,
         torch::Tensor shift_msa, torch::Tensor scale_msa, torch::Tensor gate_msa,
         torch::Tensor shift_mlp, torch::Tensor scale_mlp, torch::Tensor gate_mlp) {
  TORCH_CHECK(emb.is_cuda() && emb.dtype() == torch::kFloat32 && emb.is_contiguous(), "emb must be contiguous fp32 cuda");
  TORCH_CHECK(weight.is_contiguous() && weight.dtype() == torch::kFloat32, "weight must be contiguous fp32");
  TORCH_CHECK(bias.is_contiguous() && bias.dtype() == torch::kFloat32, "bias must be contiguous fp32");
  TORCH_CHECK(emb.dim() == 2 && emb.size(1) == 3072 && emb.size(0) >= 1, "emb must be [B, 3072]");
  TORCH_CHECK(weight.size(0) == 6 * 3072 && weight.size(1) == 3072, "weight must be [18432, 3072]");
  TORCH_CHECK(bias.numel() == 6 * 3072, "bias must be [18432]");
  const int B = (int)emb.size(0);
  torch::Tensor outs_t[6] = {shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp};
  float* outs[6];
  for (int i = 0; i < 6; ++i) {
    TORCH_CHECK(outs_t[i].is_contiguous() && outs_t[i].dtype() == torch::kFloat32 && outs_t[i].size(0) == B &&
                outs_t[i].size(1) == 3072, "bad output tensor");
    outs[i] = outs_t[i].data_ptr<float>();
  }
  const c10::cuda::OptionalCUDAGuard guard(emb.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  // Scratch through the caching allocator (torch::empty launches no kernel).
  const size_t ws_bytes = size_t(32) << 20;
  torch::Tensor tmp = torch::empty({(long)B, 6 * 3072}, emb.options());
  torch::Tensor ws = torch::empty({(long)ws_bytes}, emb.options().dtype(torch::kUInt8));
  int st = e1_launch(emb.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), outs, B,
                     tmp.data_ptr<float>(), ws.data_ptr(), ws_bytes, stream);
  TORCH_CHECK(st == 0, "e1_launch failed with status ", st);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "AdaLayerNormZero modulation extraction (cuBLASLt FAST_TF32 + bias epilogue + scatter)");
}
```

```yaml design-card
id: r1-e1-lt-fasttf32-bias-scatter
parents: [r0-adamod-dispatch-skinny-fp32-3xtf32]
operation: structural_mutation
language: cuda_cpp
niche:
  mem: tma-tensor
  st: direct
  grid: oneshot
  launch: split
  tile: rows128
  red: cta
  cache: default
  spec: all
hypothesis: >-
  One cuBLASLt FAST_TF32 GEMM (tcgen05, fp32 accumulate over full K) with the bias epilogue into tmp[B,18432], plus the
  scatter kernel, replaces every r0 path (skinny SIMT, fp32 FFMA, 3xTF32 with split pass). Under the plain-TF32 ruling
  this is the anchor at the hidden baseline's speed class. The batched pointer-array one-launch fusion (H11 second half)
  does not pay: it is slower than the big GEMM plus scatter at every B, so the scatter stays and the fusion moves to E2.
expected_effect:
  S: {pct: -58, confidence: high}
  M: {pct: -75, confidence: high}
  L: {pct: -71, confidence: high}
resources_sm100a:
  regs_per_thread: 20
  smem_per_cta_bytes: 0
  threads_per_cta: 256
  ctas_per_sm: 8
  bytes_in_flight_per_sm: 0
  launches_per_call: 2
knobs: {heuristic_index: 0, ws_mb: 32, compute: CUBLAS_COMPUTE_32F_FAST_TF32}
tests: [H11]
findings:
  - "run_tests B200 rented, 16/16 pass: B=5/16/32/96/128 -> 55.5/54.1/54.8/59.5/60.4; 131/192/211/373/384 -> 62.7/64.9/67.5/94.5/93.3; 449/512/691/773/853/919 -> 99.3/102.1/137.8/167.4/172.5/177.0 us (r0: 61/88/124/256/257 ... 576). Emulator: portal 0.537 +- 0.011 [run_tests]"
  - "max |err| vs fp64 (weight/sqrt(3072) as the harness draws, 3 trials per B): 1.39e-3 (B=5), 1.58-1.59 (16, 96), 1.75 (128), 1.53-1.67 elsewhere, max 1.752e-3 at B=128. No element exceeded atol 2e-3 + 1e-5|ref| (fraction 0 at every B). Margin to atol is thin (13%): B=128 is above the 1.7e-3 success bar [probe_b200]"
  - "cuBLASLt batched pointer-array API, accepted: A, B and D all pointer-array and no bias epilogue (7-8 heuristics, algo 73, tcgen05 kernels such as cutlass3x_sm100_tensorop_s128x128x8tf32gemm_..._2sm_epi_tma_ptr_array at B=128 and s256x256x8tf32gemm_..._2sm_ptr_array at B=919) [probe_b200]"
  - "cuBLASLt API, rejected: strided A with pointer-array C/D gives no heuristic (CUBLAS_STATUS_INVALID_VALUE, status 7); pointer-array with CUBLASLT_EPILOGUE_BIAS gives CUBLAS_STATUS_NOT_SUPPORTED (15) [probe_b200]"
  - "workaround that works: pointer-array C = six pointers into bias (C layout ld=0, i.e. a column-broadcast), beta=1: same error as the bias epilogue (1.4-1.6e-3). ldc=3072 reads out of range and gives NaN [probe_b200]"
  - "pointer arrays must be device-visible and change every call (the harness shifts pointers). A cudaMemcpyAsync or torch H2D copy adds a timed activity, so the probe used a pinned host buffer rewritten per call. That buffer is read by the GPU over PCIe: 163 us pinned vs 150 us with a fixed device array at B=919, 63-68 vs 58 at B=128 [probe_b200]"
  - "timing, interleaved 3 rounds, best of 7-8 heuristics, big GEMM+bias epilogue+scatter vs batched pointer-array (bias via C, pinned arrays): B=5 55.1 vs 63.4; 16 51.9 vs 61.8; 32 53.4 vs 64.7; 128 57.0 vs 81.6; 384 86.3 vs 100.9; 919 157.0 vs 178.4 us. Batched is 12-43% slower, and +14% at 919, +15% at 5: refuted by the >10% rule [probe_b200]"
  - "scatter cost (rented): about 5 us at B=5 (GEMM alone 49.5 vs 55.1 with scatter) and about 22 us at B=919 (GEMM 135); r0's 47 us scatter figure was inflated by the 453 MB scratch traffic around it [probe_b200]"
  - "cuBLASLt log (CUBLASLT_LOG_LEVEL=5): heuristic 0 is algoId 73 (sm100 tcgen05) with tile 64x64 and cluster 2x1x1 at B=5, tile 128x128 and cluster 2x1x1 at B=128 and B=384, tile 128x256 with no cluster at B=919. Evidence for H12: cuBLAS picks 2-CTA cluster shapes for B<=384 and a 128x256 1-SM tile at 919 [probe_b200]"
  - "heuristics 0-3 are within 1-3% of each other at B=5/16/32/128/384/919 (noise level), so a per-B table gains nothing; heuristic 0 is used everywhere [probe_b200]"
  - "dead end: profiler kernel names are not retrievable inside the probe after the first torch.profiler call; use CUBLASLT_LOG_LEVEL=5 instead [probe_b200]"
paths:
  - {max_tokens: null, lang: cuda, width: 128, threads: 256, rows: 128, grid: oneshot, mem: tma, launches: 2, x: none, w: none, st: none, cluster: 2}
runs_on:
  H200: {runs: true, representative: false, note: "cuBLASLt picks wgmma kernels on sm_90; correctness only"}
  A100: {runs: true, representative: false}
  L40S: {runs: true, representative: false}
risks:
  - "TF32 max error 1.75e-3 at B=128 vs atol 2e-3: safe on the 99% matched-ratio rule (0 elements exceeded atol in 3 trials per B) but thin; must be reported to the operator"
  - "cuBLASLt heuristic choice per B is deterministic on the rented cuBLAS 13 but depends on the portal's library; a different pick changes time, not correctness"
  - "two launches: the scatter adds about 5 us at small B (about 9% of the S band); only a fused epilogue (E2) removes it"
  - "ld_flags -lcublasLt; if the portal link line already provides cublasLt this is harmless"
measure_first: [B200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```