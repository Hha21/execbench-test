Tests done: the cuBLASLt fused kernel passes 16/16, and the two CUTLASS tcgen05 variants passed too (2SM is 5% faster at L but 2× slower at S). I have what I need; writing up the card, ledger and the first kernel now.

The first kernel is a single cuBLASLt matmul with the residual fused as the C operand (C ≠ D). It passes all 16 workloads on the rented B200 at 8.7–77.8 µs, against 12.3–90 µs for the reference.

```markdown card
# Problem card: L1 #30 `030_attention_output_projection_with_residual`

Definition from `definition.json` (Jamba-Reasoning-3B o_proj). Measurements from the rented B200 (unlocked, SM 1965 MHz;
the portal locks 1500 MHz) on 2026-10-09. No portal result yet, so Tb and Tsol are unknown.

## 1. Semantics

Reference (verbatim):

```python
@torch.no_grad()
def run(attn_output, residual, o_proj_weight):
    projected = torch.matmul(attn_output, o_proj_weight.t())
    return projected + residual
```

- DPS signature: `run(attn_output, residual, o_proj_weight, output)`.
- `attn_output`, `residual`, `output`: `[B, S, 2560]` bf16 contiguous. `o_proj_weight`: `[2560, 2560]` bf16.
- View the activations as `[M, 2560]` with `M = B·S` tokens. `D = A · Wᵀ + R`, a "TN" GEMM with M×N×K = M×2560×2560
  plus an elementwise residual: exactly `cublasLt`'s `D = α·op(A)·op(B) + β·C` with `C = residual`, `D = output`,
  α = β = 1, or CUTLASS's linear-combination epilogue with a TMA-loaded C.
- Only `M` varies; the three `M = 4096` and four `M = 8192` shapes time identically. 16 workloads = 10 sizes.
- N = 2560 = 10 × 256 = 20 × 128; K = 2560 = 40 × 64. M is a multiple of 128 except 586 (2×293), 1571, 4106 (2×2053)
  and 7976 (8×997), so M-tiles need a residue mask (TMA zero-fills out-of-bounds rows for free).

## 2. Numerics and tolerance

- Tolerances per workload: `max_rtol = 0.05`, `max_atol = 0.02` (0.0093 for 4×128), matched ratio 0.99 [workload.jsonl].
- Inputs are N(0,1) bf16, so the dot product over K = 2560 has std ≈ 50.6 and |D| reaches ≈ 250. The bf16 output
  rounding is 2⁻⁸ relative (≈ 0.2 at |D| = 50), inside rtol 0.05 by a factor of 25. fp32 accumulation in any order,
  then one bf16 rounding, is the correct precision: bf16 tensor-core MMA with fp32 accumulate is what the reference does.
- Fused residual: adding `R` in fp32 before the single bf16 rounding is slightly more accurate than the reference's
  two roundings; both pass.
- What would fail: fp16 accumulate, or bf16 accumulation across K (error ≈ 2⁻⁸·√K·|D|); reduced-precision reductions
  (`allow_bf16_reduced_precision_reduction` style split-K accumulating in bf16) are to be avoided.
- The loop's lint regex flags the literal bf16 type name in CUTLASS sources; the dtype is bf16 by problem definition
  (the cuBLASLt source avoids the literal by using `CUDA_R_16BF`).

## 3. The 16 workloads (sorted by size)

Bytes = W (13.11 MB, read once) + 3 × M × 5120 B (A, R read; D written). FLOPs = 2·M·2560². `cmp2.25` = FLOPs at the
datasheet dense bf16 peak of 2.25 PFLOP/s; `cmp1.47` = at the practical peak measured here scaled to the 1500 MHz lock
(1.93 PFLOP/s at 1965 MHz × 1500/1965) [INFERRED]. `mem8` = bytes at 8 TB/s. Floor/ref columns are the rented-B200
measurements from the task brief; `lt` is our first kernel (cuBLASLt fused, run_tests).

| M = B·S | shapes | MB | GFLOP | cmp2.25 µs | cmp1.47 µs | mem8 µs | stream floor µs | reference µs | lt µs | band |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 1×256 | 17.0 | 3.36 | 1.5 | 2.3 | 2.1 | 6.05 | 12.3 | 8.7 | S |
| 512 | 4×128 | 21.0 | 6.71 | 3.0 | 4.6 | 2.6 | 6.91 | 14.8 | 10.4 | S |
| 586 | 2×293 | 22.1 | 7.68 | 3.4 | 5.2 | 2.8 | 6.98 | 17.1 | 12.2 | S |
| 1024 | 1×1024 | 28.8 | 13.4 | 6.0 | 9.1 | 3.6 | 8.22 | 17.8 | 13.4 | S |
| 1571 | 1×1571 | 37.2 | 20.6 | 9.2 | 14.0 | 4.7 | 9.68 | 25.4 | 19.6 | S |
| 2048 | 16×128, 1×2048 | 44.6 | 26.8 | 11.9 | 18.2 | 5.6 | 11.0 | 28.2 | 23.3 | M |
| 4096 | 4×1024, 16×256, 8×512 | 76.0 | 53.7 | 23.9 | 36.5 | 9.5 | 17.7 | 48.1 | 39.3 | M |
| 4106 | 2×2053 | 76.2 | 53.8 | 23.9 | 36.6 | 9.5 | 17.9 | 48.4 | 39.1 | L |
| 7976 | 8×997 | 135.6 | 104.5 | 46.4 | 71.1 | 17.0 | 29.8 | 89.8 | 76.9 | L |
| 8192 | 16×512, 8×1024, 64×128, 32×256 | 138.9 | 107.4 | 47.7 | 73.0 | 17.4 | 30.3 | 90.2 | 77.5 | L |

## 4. Bounds and what dominates per band

- Arithmetic intensity grows with M: FLOPs/bytes = 197 at M = 256, 773 at 8192. The B200 ridge at the datasheet peak is
  281 FLOP/B, so on paper only M = 256 is memory-bound. In practice the tensor pipe reaches ≈ 1.93 PFLOP/s at
  1965 MHz (cuBLAS, 5 full waves) and the SM clock lock costs another 24%, so **every size from M ≈ 400 up is
  compute-bound** at the portal, and the L band is pure tensor-core throughput plus wave quantisation.
- S band (M ≤ 1571): the harness floor is 6–10 µs even for a plain copy (17 MB copy = 8.2 µs under cold-L2 CUPTI
  timing here, versus 2.1 µs at 8 TB/s), and cuBLAS sits at 1.4–2.0× that. The GEMM here has few output tiles
  (2–13 M-tiles × 10–20 N-tiles) and 40 serial K-steps per tile, so it is a latency chain: cuBLAS at M = 256 takes
  4.9/6.2/8.1 µs for K = 640/1280/2560. Lever: split the K loop across all 148 SMs.
- M band (2048–4096): 2048 → 80 pair-tiles of 256×256 (1.1 waves of 74 pairs), 4096 → 160 (2.2 waves). Wave
  quantisation and the fixed cost both matter.
- L band (7976–8192): 320 pair-tiles / 74 pairs = 4.32 waves, so a plain 256×256 tiling runs 5 waves at 86% use.
  Measured: cuBLAS 1.64 PFLOP/s at M = 8192 versus 1.88 at M = 9472 (exactly 5 waves) and 1.93 at 18944. **About 13%
  is on the table from stream-K or a tile that divides the work evenly.** Memory traffic (139 MB, 17 µs) is fully
  hidden by compute.
- SOLAR guess: Tsol = max(FLOPs/2.25 PF, bytes/8 TB/s) → 2.1 µs at M = 256, 47.7 µs at 8192. If so, L-band scores are
  capped by the unreachable datasheet peak, as #38's were by the 0.53× floor. Confirmed only by the first portal result.

## 5. Where the reference loses and what a fast kernel must do

- The reference is two kernels: a cuBLAS `nvjet` GEMM (65–72 µs at L, 8.2 at S) and a separate add that reads
  `projected` and `residual` and writes `output` (3 × M × 5 KB): 4 µs at S, 12–18 µs at L. `torch.addmm(..., out=)`
  is no better: a DtoD memcpy of the residual and then the `badd` GEMM variant (same time as the reference).
- A fast kernel: one launch; the residual read in the epilogue (TMA-loaded C tile) and the fp32 accumulator + C rounded
  once to bf16; per-size tiling (small tiles or split-K at S, 2SM 256×256 at L); a scheduler without a partial last
  wave at L; 2-CTA `tcgen05.mma` (cta_group::2) so each SM pair shares B-tile loads.
- Measured with CUTLASS 4.4.1 collective builders (fused residual, persistent scheduler, correct 16/16):
  1SM 128×256×64 → 25 µs at S, 95 at L; 2SM 256×256×64 cluster 2×1 → 18.9 at S, 73.7 at L. The 2SM kernel beats
  cuBLASLt by 5% at L and loses 2.2× at S (20 clusters of 40 serial K-steps). cuBLASLt wins S and M.

## 6. Design priorities

1. **S band: split/stream-K across all SMs.** cuBLAS already uses 80×64 tiles with 2-CTA at M = 256 yet takes 8.7 µs
   against a 6 µs copy floor. A custom kernel with K split 4–8 ways and an in-cluster (DSMEM) or fp32-atomic reduction
   into a single epilogue could reach ≈ 6.5–7 µs. Worth ≈ 20% on 5 workloads.
2. **L band: fix wave quantisation.** Stream-K (CUTLASS `StreamKScheduler`, needs a workspace memset in the window) or
   a 2SM tile whose count divides 74 pairs (e.g. 256×128 → 640 tiles = 8.65 waves, 96% use; 256×160 → 512 tiles,
   6.9 waves) on top of the 2SM 256×256 kernel: target 65 µs here (≈ 85 at the lock) versus 77.5 for cuBLASLt.
3. **M band:** dispatch by M between the S and L specialists; 2048 tokens (1.08 waves) is the worst quantisation case
   and wants stream-K most.
4. Cache hints: `evict_last` on D stores gained 8–9% at M sizes in #38 because of the dirty-L2 flush; try the same on
   the TMA store (`CUTLASS` epilogue has a cache-hint hook) once a custom kernel exists.
5. Keep cuBLASLt as the fallback path in any dispatcher; it is the strongest S/M kernel we have.

## 7. Results so far (rented B200, run_tests, µs)

| kernel | 256 | 512 | 586 | 1024 | 1571 | 2048 | 4096 | 4106 | 7976 | 8192 | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference (matmul + add) | 12.3 | 14.8 | 17.1 | 17.8 | 25.4 | 28.2 | 48.1 | 48.4 | 89.8 | 90.2 | – |
| **oproj-cublaslt-fused (this submission)** | 8.7 | 10.4 | 12.2 | 13.4 | 19.6 | 23.3 | 39.3 | 39.1 | 76.9 | 77.5 | 16/16 |
| CUTLASS 1SM 128×256×64 persistent | 25.1 | 25.2 | 25.9 | 26.0 | 26.2 | 43.0 | 61.0 | 61.6 | 94.6 | 95.0 | 16/16 |
| CUTLASS 2SM 256×256×64, cluster 2×1 | 18.9 | 19.0 | 19.2 | 19.9 | 21.3 | 32.9 | 48.0 | 48.0 | 73.1 | 73.7 | 16/16 |

## 8. Open questions

- Hidden Tb per workload: is it the reference (no PyTorch-only way to fuse the residual), or something smarter (e.g.
  torch.compile)? The first portal result answers this.
- Does SOLAR use 2.25 PFLOP/s dense bf16 for the compute term? If so the L band is capped near 0.6 even at the
  practical peak.
- How do cuBLAS's heuristics behave at the 1500 MHz lock (same algorithms, compute terms 1.31× slower)?
- Can a split-K kernel at S reduce inside one launch without a second kernel (cluster DSMEM reduction versus fp32
  atomics into a workspace plus an ordered final epilogue)?
- Does the dirty-L2 flush (126 MB) cost the same ≈ 6 µs fixed here as the 17 MB copy suggests, and does `evict_last`
  on D stores recover part of it?
- Is there a CUTLASS stream-K configuration for sm100 2SM kernels that does not add a memset or reduce kernel to the
  timed window?
```

```yaml ledger
# Hypothesis ledger for #30 (030_attention_output_projection_with_residual). Seeded 2026-10-09 by the research session.
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.
constraints:
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement,
  not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the
  timed window.'
- 'Clusters only at <= 300 tokens (H16): clustered large grids lost 2-5% on the portal and seemed to slow later plain launches.'
- 'Eviction-priority operations (createpolicy, applypriority.L2::evict_normal) only on the current call''s own input/output
  buffers, inside our kernel. Any candidate that uses applypriority goes to the operator for a legitimacy check before a
  portal slot (r13, H21).'
hypotheses:
- id: H1
  statement: Fusing the residual as the GEMM's C operand (C != D) removes the reference's separate add kernel at no cost.
  status: supported
  evidence:
  - 'run_tests: cublasLt fused 8.7..77.5 us vs reference 12.3..90.2 us (matmul alone 8.2..72 us), 16/16 pass'
  - 'probe: torch.addmm(out=) launches a DtoD memcpy of the residual plus the badd nvjet kernel; same time as the reference'
- id: H2
  statement: The L band is tensor-core bound and loses about 13% to wave quantisation (320 pair tiles of 256x256 over 74 SM pairs
    = 4.32 waves); stream-K or a tile count that divides evenly recovers most of it.
  status: open
  evidence:
  - 'probe: cuBLAS 1.64 PFLOP/s at M=8192 vs 1.88 at M=9472 (5 full waves) and 1.93 at M=18944 (rented B200, 1965 MHz)'
  - 'probe: K=2560 vs 5120 vs 10240 at M=8192 gives the same TFLOP/s, so per-tile prologue/epilogue is not the limit'
- id: H3
  statement: The S band (M <= 1571) is a latency chain of 40 serial K-steps on few tiles; splitting K over all 148 SMs with an
    in-kernel reduction approaches the 6-8 us harness copy floor.
  status: open
  evidence:
  - 'probe: cuBLAS at M=256 takes 4.9/6.2/8.1 us for K=640/1280/2560; a 17 MB copy takes 6.1-8.2 us under harness timing'
  - 'run_tests: CUTLASS 128x256 (20 CTAs) 25 us and 2SM 256x256 (20 clusters) 18.9 us at M=256 vs cublasLt 8.7 us'
- id: H4
  statement: 2-CTA tcgen05 MMA (cta_group::2, 256x256x64 tiles) beats 1SM 128x256x64 at every size.
  status: supported
  evidence:
  - 'run_tests: CUTLASS 2SM 18.9/32.9/48.0/73.7 us vs 1SM 25.1/43.0/61.0/95.0 us at M=256/2048/4096/8192'
- id: H5
  statement: A per-size dispatcher (cublasLt or split-K specialist at S/M, CUTLASS 2SM stream-K at L) beats cublasLt alone.
  status: open
  evidence:
  - 'run_tests: CUTLASS 2SM 256x256 is already 5% faster than cublasLt at M=7976-8192 (73.1-73.7 vs 76.9-77.5 us)'
- id: H6
  statement: The datasheet 2.25 PFLOP/s dense bf16 peak is unreachable at the 1500 MHz lock (practical ~1.47 PFLOP/s), so if
    SOLAR uses it the L-band score is capped near 0.6 and S-band workloads hold most of the headroom.
  status: open
  evidence:
  - 'probe: best cuBLAS rate 1.93 PFLOP/s at 1965 MHz with no power throttling (300 W of 1000 W); scales to 1.47 at 1500 MHz'
- id: H7
  statement: The hidden baseline Tb is close to the reference time (PyTorch-only code cannot fuse the residual into cuBLAS).
  status: open
  evidence:
  - 'probe: matmul+add and addmm(out=) are within 3% of each other at every size; the first portal result decides'
- id: H8
  statement: evict_last on the output stores (TMA store cache hint) recovers part of the dirty-L2 flush cost at S/M, as in #38.
  status: open
  evidence:
  - '#38 portal: r5 0.609 vs r3 0.588 (M -8%); untested here, the first kernel is a library call'
- id: H9
  statement: cuBLAS heuristics pick the same algorithms at the portal's locked clock, so the first kernel's S/L ratio on the
    portal follows the compute/memory split (L slows ~1.3x, S less).
  status: open
  evidence:
  - 'none yet; portal submission 1'
```

### Rationale

The first kernel is deliberately a library call: one `cublasLtMatmul` with the residual as the C operand, so the add is fused and a single kernel runs. It passed all 16 workloads and is faster than the reference everywhere, which is enough to unlock Tb and Tsol. Custom CUTLASS tcgen05 variants were built and measured for the card (2SM already wins at L) but are left for the design rounds.

```json solution-spec
{"name": "oproj-cublaslt-fused-v0", "definition": "030_attention_output_projection_with_residual", "author": "solx-loop",
 "description": "One cublasLt matmul with C=residual, D=output (beta=1): o_proj GEMM with the residual add fused in the epilogue, bf16 in/out, fp32 accumulate.",
 "spec": {"languages": ["cublas"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run",
          "dependencies": ["torch"], "destination_passing_style": true,
          "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"], "ld_flags": ["-lcublasLt"]}}}
```

```cuda file=kernel.cu
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <mutex>

// D[M,N] = A[M,K] * W[N,K]^T + R[M,N], all bf16 row-major, fp32 accumulate, one cublasLt matmul
// with C = residual and D = output (C != D is allowed by cublasLt, so the residual add is fused).
// cuBLAS is column-major: D^T[N,M] = op_T(W_col)[N,K] * op_N(A_col)[K,M] + R^T.

namespace {
cublasLtHandle_t get_handle() {
    static cublasLtHandle_t h = nullptr;
    static std::once_flag once;
    std::call_once(once, [] { cublasLtCreate(&h); });
    return h;
}
}  // namespace

extern "C" int oproj_launch(const void* A, const void* R, const void* W, void* D,
                             int M, int N, int K, void* ws, size_t ws_bytes, cudaStream_t stream) {
    cublasLtHandle_t handle = get_handle();
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr, ld = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasStatus_t st;
#define LT(x) st = (x); if (st != CUBLAS_STATUS_SUCCESS) goto done;
    LT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    {
        cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
        LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
        LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
    }
    // "A" operand = W viewed column-major: K rows x N cols, ld = K; transposed by the op.
    LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, K));
    // "B" operand = attn viewed column-major: K rows x M cols, ld = K.
    LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, M, K));
    // C (residual) and D (output) viewed column-major: N rows x M cols, ld = N.
    LT(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M, N));
    LT(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, N, M, N));
    LT(cublasLtMatmulPreferenceCreate(&pref));
    LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));
    {
        cublasLtMatmulHeuristicResult_t heur{};
        int nret = 0;
        LT(cublasLtMatmulAlgoGetHeuristic(handle, op, la, lb, lc, ld, pref, 1, &heur, &nret));
        if (nret == 0) { st = CUBLAS_STATUS_NOT_SUPPORTED; goto done; }
        const float alpha = 1.0f, beta = 1.0f;
        LT(cublasLtMatmul(handle, op, &alpha, W, la, A, lb, &beta, R, lc, D, ld,
                          &heur.algo, ws, ws_bytes, stream));
    }
done:
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (ld) cublasLtMatrixLayoutDestroy(ld);
    if (lc) cublasLtMatrixLayoutDestroy(lc);
    if (lb) cublasLtMatrixLayoutDestroy(lb);
    if (la) cublasLtMatrixLayoutDestroy(la);
    if (op) cublasLtMatmulDescDestroy(op);
    return (int)st;
#undef LT
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

extern "C" int oproj_launch(const void* A, const void* R, const void* W, void* D,
                             int M, int N, int K, void* ws, size_t ws_bytes, cudaStream_t stream);

namespace {
constexpr int64_t kWorkspace = 32ll << 20;  // 32 MB, the size cuBLAS recommends on Hopper/Blackwell
}

void run(torch::Tensor attn_output, torch::Tensor residual, torch::Tensor o_proj_weight, torch::Tensor output) {
    TORCH_CHECK(attn_output.is_cuda() && residual.is_cuda() && o_proj_weight.is_cuda() && output.is_cuda());
    TORCH_CHECK(attn_output.scalar_type() == at::kBFloat16 && residual.scalar_type() == at::kBFloat16 &&
                o_proj_weight.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16);
    TORCH_CHECK(attn_output.is_contiguous() && residual.is_contiguous() && o_proj_weight.is_contiguous() && output.is_contiguous());
    const int64_t K = attn_output.size(-1);
    const int64_t N = o_proj_weight.size(0);
    const int64_t M = attn_output.numel() / K;
    TORCH_CHECK(o_proj_weight.size(1) == K && residual.numel() == M * N && output.numel() == M * N);
    TORCH_CHECK(M <= INT32_MAX && (K % 8) == 0 && (N % 8) == 0);

    const c10::cuda::OptionalCUDAGuard guard(attn_output.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // Workspace from the caching allocator: no kernel is launched by this.
    auto ws = at::empty({kWorkspace}, attn_output.options().dtype(at::kByte));
    int st = oproj_launch(attn_output.data_ptr(), residual.data_ptr(), o_proj_weight.data_ptr(), output.data_ptr(),
                          (int)M, (int)N, (int)K, ws.data_ptr(), (size_t)kWorkspace, stream);
    TORCH_CHECK(st == 0, "cublasLt matmul failed with status ", st);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "o_proj + residual via cublasLt (bf16 in, fp32 accumulate)");
}
```

```yaml design-card
id: oproj-cublaslt-fused-v0
parents: []
operation: new_design
language: cuda_cpp
niche:
  mem: tma-tensor
  st: direct
  grid: persistent
  launch: fused
  tile: rows128
  red: cta
  cache: default
  spec: all
hypothesis: >-
  One cublasLt matmul with the residual passed as the C operand and the output as D fuses the
  residual add into the GEMM epilogue, so a single kernel replaces the reference's GEMM plus add;
  cuBLAS's per-shape heuristics (small 2-CTA tiles at small M) are the best S/M kernel available today.
expected_effect:
  S: {pct: -28, confidence: medium}
  M: {pct: -18, confidence: medium}
  L: {pct: -14, confidence: medium}
resources_sm100a:
  regs_per_thread: 0
  smem_per_cta_bytes: 0
  threads_per_cta: 0
  ctas_per_sm: 0
  bytes_in_flight_per_sm: 0
  launches_per_call: 1
knobs: {workspace_mb: 32, algo: heuristic-0, alpha: 1, beta: 1}
tests: [H1, H7, H9]
findings:
  - "cuBLAS matmul alone (rented B200, 1965 MHz): 8.2 us at 256 tokens (0.41 PFLOP/s) to 65-72 us at 8192 (1.5-1.64 PFLOP/s) [probe_b200]"
  - "reference = cuBLAS nvjet GEMM + separate add; the add costs 4 us at S and 12-18 us at L [probe_b200]"
  - "torch.addmm(out=) launches a DtoD memcpy of the residual then the badd nvjet kernel; same time as the reference [probe_b200]"
  - "cublasLt with C=residual, D=output: one kernel, 16/16 pass, 8.7 (256 tok) .. 77.8 us (8192); 1.4x the copy floor at S, 2.5x at L [run_tests]"
  - "CUTLASS 4.4 sm100 1SM 128x256x64 persistent + TMA C/D epilogue: compiles in 25 s, 198 regs, correct; 25 us at S (20 CTAs), 95 us at L [run_tests]"
  - "CUTLASS 2SM 256x256x64 cluster 2x1: 18.9 us at S, 73.7 us at L, 5% faster than cublasLt at L and 2.2x slower at S [run_tests]"
  - "tensor peak: cuBLAS reaches 1.93 PFLOP/s at M=18944 (full waves) but 1.64 at M=8192 (4.32 waves): L loses ~13% to wave quantisation [probe_b200]"
  - "K=2560 vs 5120 vs 10240 at M=8192 gives the same TFLOP/s: per-tile prologue/epilogue is not the L-band limit [probe_b200]"
  - "M=256: cuBLAS 4.9/6.2/8.1 us at K=640/1280/2560: serial K-steps dominate S; split-K over 148 SMs is the S lever [probe_b200]"
  - "a 17 MB bf16 copy in one kernel takes 8.2 us under harness-style cold-L2 timing (2.1 us at 8 TB/s): ~6 us fixed cost at S [probe_b200]"
  - "no throttling: 1965 MHz and 300 W of 1000 W during sustained GEMM [probe_b200]"
  - "dead end for the compile tool: the lint regex rejects the literal bf16 type name in CUTLASS sources although the dtype is bf16 by definition; cublasLt source uses CUDA_R_16BF [compile_b200]"
  - "a .cpp-only solution does not get the CUDA include path; keep a kernel.cu in the sources [run_tests]"
paths:
  - {max_tokens: null, lang: cuda, width: 128, threads: 256, rows: 128, grid: persistent, mem: tma, wstat: false, launches: 1, prefetch: false, x: none, w: none, st: none}
runs_on:
  H200: {runs: true, representative: true, note: "cublasLt picks Hopper kernels; same fused structure"}
  A100: {runs: true, representative: false, note: "correctness only; cuBLAS Ampere kernels"}
  L40S: {runs: true, representative: false}
risks:
  - "cuBLAS heuristic may choose a split-K algorithm with a reduce kernel or workspace memset on some shapes (consistent per shape, so CUPTI matching still holds)"
  - "link against -lcublasLt relies on the toolkit's libcublasLt in /usr/local/cuda/lib64 on the portal image"
  - "per-call cublasLt descriptor creation and heuristic query are host-only; not timed by CUPTI"
measure_first: [H200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```

Recap: this is a bf16 GEMM (M×2560×2560) with a fused residual, compute-bound above about 400 tokens and latency-bound below. The reference loses 15–20% to a separate add kernel. The submitted first kernel is a single cuBLASLt matmul with the residual as the C operand, correct on all 16 workloads and faster than the reference at every size. Next rounds should target wave quantisation at L with a CUTLASS 2SM stream-K kernel (already 5% ahead of cuBLASLt there) and a split-K design for the S band, then a dispatcher by token count.