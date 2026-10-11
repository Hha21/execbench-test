### Rationale
48-k stages (an SW128 32-k box plus an SW64 16-k box per operand, 12 UMMAs per commit, 3 stages) were built and were numerically correct on the first try. On the rented card they cut the cycles that matter wherever MMA work per 32-k stage is below the ~900-clk floor: Bpad 384 -3..-7.5%, Ntile 352 -11..-14%, Ntile 400 -7%. With L2-hot W, the 384 stage drops to 1175 clk, 2% above the 1152-clk MMA ideal. So the 48-k stage hides the floor, and the remaining gap with real W is DRAM latency through a 3-deep ring. That gap is set in ns, so it should cost fewer cycles at the portal's 1.5 GHz. Two follow-ups failed and are recorded as dead ends: separate W and emb rings (+400 clk for each extra commit) and L2 prefetch of W (+18-20%). The candidate sends every B > 256 through one hand-written kernel and drops CUTLASS. run_tests passed 16/16; against the current best on the same GPU it is M -2.3% and L -4.7% by band, with 373 -7.8%, 449 -11.3% and 773 -8.8%.

```json solution-spec
{"name": "r5-e3-tc-k48-pair", "definition": "035_flux_ada_layer_norm_zero_modulation_extraction", "author": "solx-loop", "description": "48-k TMA/UMMA stages (SW128 32-k + SW64 16-k boxes) for the two-accumulator 2SM pair kernel at B > 256 (nb = ceil(B/512) batch tiles), hiding the per-stage floor of 32-k rings", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": [], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=tc_small.cu
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>

// B <= 256: hand-written tcgen05 TF32 GEMM, one 128-row W tile per CTA (144 CTAs, one per SM), all batch rows in
// one UMMA N = B rounded up to 16 (<= 256), fp32 accumulator in TMEM over the full K = 3072.
// W streams from DRAM once through a TMA ring with an L2 evict_first hint: the harness leaves the L2 full of dirty
// zero lines, and evict_first W lines recycle each other instead of forcing those lines to be written back inside
// the timed window. Each stage holds kbs k-blocks of 32 (128 B swizzle rows) of W and of emb (emb from L2).
// The TMA tensor maps use the TFLOAT32 type (round to nearest tf32: max |err| 1.3-1.6e-3; FLOAT32 truncates, 3.6-4.2e-3).
// Epilogue: 4 warps read one TMEM lane quadrant each (row m = W row), add bias[m] and store out_c[b * 3072 + m - c * 3072];
// c = m0 / 3072 is uniform per CTA.
// tcgen05 exists only on sm_100a: the body is compiled for the arch-specific target (as CUTLASS guards its kernels).
namespace {
constexpr int kD = 3072, kN = 6 * kD, kRows = 128, kBK = 32, kKB = kD / kBK, kWBlk = kRows * kBK * 4;
constexpr int kMaxStages = 16, kSmemDyn = 225 * 1024;
struct OutPtrs { float* p[6]; };

#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ bool try_wait(uint32_t bar, uint32_t par) {
  uint32_t ok;
  asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
               : "=r"(ok) : "r"(bar), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t par) { while (!try_wait(bar, par)) {} }
__device__ __forceinline__ void mbar_init(uint32_t bar) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar)); }
__device__ __forceinline__ void umma_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory");
}
// K-major, 128 B swizzle UMMA operand: 8-row groups 1024 B apart (SBO), version 1 (sm100), layout SWIZZLE_128B.
__device__ __forceinline__ uint64_t sw128_desc(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 46) |
         ((uint64_t)2 << 61);
}
#endif

__global__ void __launch_bounds__(128, 1) tc_small_kernel(const __grid_constant__ CUtensorMap mapW,
                                                         const __grid_constant__ CUtensorMap mapE,
                                                         const float* __restrict__ bias, OutPtrs outs, int B, int Bpad,
                                                         int kbs, int stages, int tmem_cols) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  extern __shared__ uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t full[kMaxStages], empty[kMaxStages], accum;
  __shared__ uint32_t tmem_slot;
  const uint32_t sbase = (smem_u32(smem_raw) + 1023) & ~1023u;
  const uint32_t eBlk = Bpad * 128, stageBytes = kbs * (kWBlk + eBlk);
  const int nIt = kKB / kbs;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int m0 = blockIdx.x * kRows;
  if (threadIdx.x == 0) {
    for (int i = 0; i < stages; ++i) { mbar_init(smem_u32(&full[i])); mbar_init(smem_u32(&empty[i])); }
    mbar_init(smem_u32(&accum));
    asm volatile("fence.mbarrier_init.release.cluster;");
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapE) : "memory");
  }
  if (warp == 3) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(&tmem_slot)), "r"(tmem_cols));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t tmem = tmem_slot;

  if (warp == 0 && lane == 0) {   // TMA producer
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    for (int i = 0; i < nIt; ++i) {
      const int s = i % stages;
      if (i >= stages) mbar_wait(smem_u32(&empty[s]), ((i / stages) - 1) & 1);
      const uint32_t fb = smem_u32(&full[s]);
      const uint32_t a = sbase + s * stageBytes, b = a + kbs * kWBlk;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(fb), "r"(stageBytes) : "memory");
      asm volatile(
          "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5}], [%2], %6;"
          ::"r"(a), "l"((uint64_t)&mapW), "r"(fb), "r"(0), "r"(m0), "r"(i * kbs), "l"(pol) : "memory");
      for (int j = 0; j < kbs; ++j)
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
                     ::"r"(b + j * eBlk), "l"((uint64_t)&mapE), "r"(fb), "r"((i * kbs + j) * kBK), "r"(0) : "memory");
    }
  } else if (warp == 1 && lane == 0) {   // MMA issuer: UMMA 128 x Bpad x 8, four per 32-wide k-block
    const uint32_t idesc = (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)(Bpad >> 3) << 17) | ((uint32_t)(kRows >> 4) << 24);
    for (int i = 0; i < nIt; ++i) {
      const int s = i % stages;
      mbar_wait(smem_u32(&full[s]), (i / stages) & 1);
      asm volatile("tcgen05.fence::after_thread_sync;");
      for (int j = 0; j < kbs; ++j) {
        const uint32_t a = sbase + s * stageBytes + j * kWBlk, b = sbase + s * stageBytes + kbs * kWBlk + j * eBlk;
        const uint64_t da = sw128_desc(a), db = sw128_desc(b);
#pragma unroll
        for (int k = 0; k < kBK / 8; ++k) {
          const uint32_t acc = (i | j | k) != 0;
          asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
                       ::"r"(tmem), "l"(da + 2 * k), "l"(db + 2 * k), "r"(idesc), "r"(acc));
        }
      }
      umma_commit(smem_u32(&empty[s]));
    }
    umma_commit(smem_u32(&accum));
  }
  __syncwarp();

  mbar_wait(smem_u32(&accum), 0);
  asm volatile("tcgen05.fence::after_thread_sync;");
  const int m = m0 + warp * 32 + lane;
  const int c = m0 / kD;
  const float bv = bias[m];
  float* o = outs.p[c] + (m - c * kD);
  for (int c0 = 0; c0 < Bpad; c0 += 16) {
    uint32_t r[16];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                 : "r"(tmem + ((uint32_t)(warp * 32) << 16) + c0));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
    for (int i = 0; i < 16; ++i)
      if (c0 + i < B) o[(size_t)(c0 + i) * kD] = __uint_as_float(r[i]) + bv;
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (warp == 3) {
    asm volatile("tcgen05.fence::after_thread_sync;");
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(tmem_cols));
  }
#endif
}
}  // namespace

extern "C" int adamod_tc_small(const float* emb, const float* W, const float* bias, float* const* outs, int B,
                               cudaStream_t stream) {
  if (B < 1 || B > 256) return 400;
  const int Bpad = (B + 15) / 16 * 16;
  const int tmem_cols = Bpad <= 32 ? 32 : Bpad <= 64 ? 64 : Bpad <= 128 ? 128 : 256;
  const int perK = kWBlk + Bpad * 128, avail = kSmemDyn - 1024;
  int kbs = 1;   // largest k-blocks per stage that still leaves a 3-deep ring
  for (int cand : {4, 3, 2})
    if (avail / (cand * perK) >= 3) { kbs = cand; break; }
  int stages = avail / (kbs * perK);
  if (stages > kMaxStages) stages = kMaxStages;
  CUtensorMap mW, mE;
  // W viewed as [96 k-blocks][18432 rows][32]: one box = kbs k-blocks of 128 rows, each a canonical SW128 atom stack.
  cuuint64_t dW[3] = {(cuuint64_t)kBK, (cuuint64_t)kN, (cuuint64_t)kKB}, sW[2] = {(cuuint64_t)kD * 4, (cuuint64_t)kBK * 4};
  cuuint32_t bW[3] = {kBK, kRows, (cuuint32_t)kbs}, es[3] = {1, 1, 1};
  if (cuTensorMapEncodeTiled(&mW, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 3, (void*)W, dW, sW, bW, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return 401;
  // emb rows B..Bpad-1 are out of bounds and zero-filled by TMA.
  cuuint64_t dE[2] = {(cuuint64_t)kD, (cuuint64_t)B}, sE[1] = {(cuuint64_t)kD * 4};
  cuuint32_t bE[2] = {kBK, (cuuint32_t)Bpad};
  if (cuTensorMapEncodeTiled(&mE, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 2, (void*)emb, dE, sE, bE, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return 402;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(tc_small_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemDyn);
    attr_set = true;
  }
  OutPtrs op;
  for (int i = 0; i < 6; ++i) op.p[i] = outs[i];
  tc_small_kernel<<<kN / kRows, 128, stages * kbs * perK + 1024, stream>>>(mW, mE, bias, op, B, Bpad, kbs, stages, tmem_cols);
  return (int)cudaGetLastError();
}
```

```cuda file=tc_pair.cu
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>

// 131 < B <= 256: 2SM tcgen05 TF32 GEMM. A cluster of two CTAs computes a 256-row W tile with one cta_group::2 UMMA
// (M = 256, N = Bpad <= 256, K = 8) per step: each CTA holds its own 128 W rows and half (Bpad / 2) of the emb rows in
// shared memory, and its accumulator is 128 lanes x Bpad columns of its own TMEM. Only the leader CTA issues the MMAs;
// both CTAs' TMA loads signal the leader's full barrier, and the commit is multicast to both CTAs' empty barriers.
// Stages hold kbs = 2 k-blocks of 32: the E2 probe measured 8-UMMA stages at 749 clk/k-block against 1108 for the
// 1-k-block stages of tc_small (per-stage MMA/commit chain cost of about 250-300 clk dominates at 1 k-block).
// W loads carry the L2 evict_first policy as in tc_small. Epilogue per CTA is the same as tc_small (bias + six-way split).
namespace {
constexpr int kD = 3072, kN = 6 * kD, kRows = 128, kBK = 32, kKB = kD / kBK, kWBlk = kRows * kBK * 4;
constexpr int kMaxStages = 16, kSmemDyn = 225 * 1024;
struct OutPtrs { float* p[6]; };

#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t mapa0(uint32_t a) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(0));
  return r;
}
__device__ __forceinline__ bool try_wait(uint32_t bar, uint32_t par) {
  uint32_t ok;
  asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
               : "=r"(ok) : "r"(bar), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t par) {
  uint32_t n = 0;
  while (!try_wait(bar, par))
    if (++n > (1u << 27)) __trap();   // a broken pipeline fails the call instead of hanging it
}
__device__ __forceinline__ void mbar_init(uint32_t bar) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar)); }
__device__ __forceinline__ void umma_commit2(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
               ::"r"(bar), "h"((uint16_t)3) : "memory");
}
__device__ __forceinline__ uint64_t sw128_desc(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 46) |
         ((uint64_t)2 << 61);
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
#endif

__global__ void __launch_bounds__(128, 1) tc_pair_kernel(const __grid_constant__ CUtensorMap mapW,
                                                        const __grid_constant__ CUtensorMap mapE,
                                                        const float* __restrict__ bias, OutPtrs outs, int B, int Bpad,
                                                        int kbs, int stages, int tmem_cols) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  extern __shared__ uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t full[kMaxStages], empty[kMaxStages], accum;
  __shared__ uint32_t tmem_slot;
  const uint32_t sbase = (smem_u32(smem_raw) + 1023) & ~1023u;
  const int ebrows = Bpad / 2;
  const uint32_t eBlk = ebrows * 128, stageBytes = kbs * (kWBlk + eBlk);   // per CTA
  const int nIt = kKB / kbs;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const int m0 = blockIdx.x * kRows;
  if (threadIdx.x == 0) {
    for (int i = 0; i < stages; ++i) { mbar_init(smem_u32(&full[i])); mbar_init(smem_u32(&empty[i])); }
    mbar_init(smem_u32(&accum));
    asm volatile("fence.mbarrier_init.release.cluster;");
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapE) : "memory");
  }
  if (warp == 3) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(&tmem_slot)), "r"(tmem_cols));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  cluster_sync();
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t tmem = tmem_slot;

  if (warp == 0 && lane == 0) {   // TMA producer (both CTAs)
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    for (int i = 0; i < nIt; ++i) {
      const int s = i % stages;
      if (i >= stages) mbar_wait(smem_u32(&empty[s]), ((i / stages) - 1) & 1);
      const uint32_t fl = smem_u32(&full[s]);
      const uint32_t fb = mapa0(fl);   // the leader's full barrier collects the bytes of both CTAs
      if (rank == 0)
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(fl), "r"(2 * stageBytes) : "memory");
      const uint32_t a = sbase + s * stageBytes, b = a + kbs * kWBlk;
      asm volatile(
          "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5}], [%2], %6;"
          ::"r"(a), "l"((uint64_t)&mapW), "r"(fb), "r"(0), "r"(m0), "r"(i * kbs), "l"(pol) : "memory");
      for (int j = 0; j < kbs; ++j)
        asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
                     ::"r"(b + j * eBlk), "l"((uint64_t)&mapE), "r"(fb), "r"((i * kbs + j) * kBK), "r"((int)rank * ebrows) : "memory");
    }
  } else if (warp == 1 && lane == 0 && rank == 0) {   // MMA issuer (leader): UMMA 256 x Bpad x 8, four per k-block
    const uint32_t idesc = (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)(Bpad >> 3) << 17) | ((uint32_t)(2 * kRows >> 4) << 24);
    for (int i = 0; i < nIt; ++i) {
      const int s = i % stages;
      mbar_wait(smem_u32(&full[s]), (i / stages) & 1);
      asm volatile("tcgen05.fence::after_thread_sync;");
      for (int j = 0; j < kbs; ++j) {
        const uint32_t a = sbase + s * stageBytes + j * kWBlk, b = sbase + s * stageBytes + kbs * kWBlk + j * eBlk;
        const uint64_t da = sw128_desc(a), db = sw128_desc(b);
#pragma unroll
        for (int k = 0; k < kBK / 8; ++k) {
          const uint32_t acc = (i | j | k) != 0;
          asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, p; }"
                       ::"r"(tmem), "l"(da + 2 * k), "l"(db + 2 * k), "r"(idesc), "r"(acc));
        }
      }
      umma_commit2(smem_u32(&empty[s]));
    }
    umma_commit2(smem_u32(&accum));
  }
  __syncwarp();

  mbar_wait(smem_u32(&accum), 0);
  asm volatile("tcgen05.fence::after_thread_sync;");
  const int m = m0 + warp * 32 + lane;
  const int c = m0 / kD;
  const float bv = bias[m];
  float* o = outs.p[c] + (m - c * kD);
  for (int c0 = 0; c0 < Bpad; c0 += 16) {
    uint32_t r[16];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                 : "r"(tmem + ((uint32_t)(warp * 32) << 16) + c0));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
    for (int i = 0; i < 16; ++i)
      if (c0 + i < B) o[(size_t)(c0 + i) * kD] = __uint_as_float(r[i]) + bv;
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  cluster_sync();
  if (warp == 3) {
    asm volatile("tcgen05.fence::after_thread_sync;");
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(tmem_cols));
  }
#endif
}
}  // namespace

extern "C" int adamod_tc_pair(const float* emb, const float* W, const float* bias, float* const* outs, int B,
                              cudaStream_t stream) {
  if (B < 1 || B > 256) return 410;
  const int Bpad = (B + 15) / 16 * 16;
  const int tmem_cols = Bpad <= 32 ? 32 : Bpad <= 64 ? 64 : Bpad <= 128 ? 128 : 256;
  const int ebrows = Bpad / 2;
  const int perK = kWBlk + ebrows * 128, avail = kSmemDyn - 1024;
  const int kbs = 2;
  int stages = avail / (kbs * perK);
  if (stages > kMaxStages) stages = kMaxStages;
  CUtensorMap mW, mE;
  cuuint64_t dW[3] = {(cuuint64_t)kBK, (cuuint64_t)kN, (cuuint64_t)kKB}, sW[2] = {(cuuint64_t)kD * 4, (cuuint64_t)kBK * 4};
  cuuint32_t bW[3] = {kBK, kRows, (cuuint32_t)kbs}, es[3] = {1, 1, 1};
  if (cuTensorMapEncodeTiled(&mW, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 3, (void*)W, dW, sW, bW, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return 411;
  // Each CTA loads ebrows emb rows (rank * ebrows ...); rows >= B are out of bounds and zero-filled by TMA.
  cuuint64_t dE[2] = {(cuuint64_t)kD, (cuuint64_t)B}, sE[1] = {(cuuint64_t)kD * 4};
  cuuint32_t bE[2] = {kBK, (cuuint32_t)ebrows};
  if (cuTensorMapEncodeTiled(&mE, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 2, (void*)emb, dE, sE, bE, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return 412;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(tc_pair_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemDyn);
    attr_set = true;
  }
  OutPtrs op;
  for (int i = 0; i < 6; ++i) op.p[i] = outs[i];
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(kN / kRows);
  cfg.blockDim = dim3(128);
  cfg.dynamicSmemBytes = stages * kbs * perK + 1024;
  cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 2;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, tc_pair_kernel, mW, mE, bias, op, B, Bpad, kbs, stages, tmem_cols);
  return (int)cudaGetLastError();
}
```

```cuda file=tc_k48.cu
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>

// B > 256: 2SM tcgen05 TF32 GEMM with two TMEM accumulators and nb = ceil(B / 512) batch tiles per 256-row W tile.
// Ntile = roundup(ceil(B / nb), 16) in (256, 512]: one pass over K computes N1 = 256 columns into TMEM columns
// [0, 256) and N2 = Ntile - 256 columns into [256, Ntile), both from the same W stage.
// K stage KS = 48 (Ntile <= 432) or 32 (Ntile > 432). A 48-k stage is a 32-k box with 128 B swizzle plus a 16-k box
// with 64 B swizzle per operand (two TMA maps per operand, two UMMA descriptor families: SW128 advanced by 32 B for
// K-steps 0-3, SW64 with SBO 512 B advanced by 32 B for K-steps 4-5), 12 UMMAs and one commit per stage, 64 stages.
// It carries 1.5x the MMA work of a 32-k stage, which hides the ~900-1200 clk per-stage floor of 32-k rings below
// N ~ 470 (E3 probe: Bpad 384 864 clk per 32-k vs 934; L2-hot W 783 = MMA-bound; Ntile 352 780 vs 891).
// Stage layout per CTA (1024-aligned): [W 32-k 16 KB][emb 32-k half x 128 B][W 16-k 8 KB][emb 16-k half x 64 B].
// Each CTA of the pair loads Ntile / 2 contiguous emb rows (n0 + rank * Ntile / 2 ...); the UMMA B operand of N columns
// takes N / 2 rows from each CTA, so accumulator columns map to emb rows as
//   j < 128 -> j, 128..255 -> half + j - 128, 256..256+h2 -> j - 128, rest -> half + j - 128 - h2.
// Grid: 72 W tiles x nb batch tiles clusters of 2; cluster id -> (W tile = id / nb, batch tile = id % nb), so the batch
// tiles of a W tile run together and the second read of W hits L2. W loads carry L2 evict_first (evict_normal costs
// +15% at Bpad 384 with 48-k stages); TMA maps are TFLOAT32 (round on load); emb rows >= B are zero-filled by TMA and
// the epilogue skips them. The epilogue remaps once per 8 columns (a per-column remap spills r[] to local memory).
namespace {
constexpr int kD = 3072, kN = 6 * kD, kRows = 128, kWTiles = kN / (2 * kRows);
constexpr int kW128 = kRows * 128, kW64 = kRows * 64;   // per-CTA W box bytes: 32-k (SW128) and 16-k (SW64)
constexpr int kMaxStages = 8, kSmemDyn = 226 * 1024, kThreads = 256;
struct OutPtrs { float* p[6]; };

#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t mapa0(uint32_t a) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(0));
  return r;
}
__device__ __forceinline__ bool try_wait(uint32_t bar, uint32_t par) {
  uint32_t ok;
  asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
               : "=r"(ok) : "r"(bar), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t par) {
  uint32_t n = 0;
  while (!try_wait(bar, par))
    if (++n > (1u << 27)) __trap();   // a broken pipeline fails the call instead of hanging it
}
__device__ __forceinline__ void mbar_init(uint32_t bar) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar)); }
__device__ __forceinline__ void umma_commit2(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
               ::"r"(bar), "h"((uint16_t)3) : "memory");
}
// K-major UMMA operands, version 1 (sm100): SW128 = 8-row groups 1024 B apart, layout 2; SW64 = 512 B apart, layout 4.
__device__ __forceinline__ uint64_t sw128_desc(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 46) |
         ((uint64_t)2 << 61);
}
__device__ __forceinline__ uint64_t sw64_desc(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(512 >> 4) << 32) | ((uint64_t)1 << 46) |
         ((uint64_t)4 << 61);
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void mma2(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, p; }"
               ::"r"(d), "l"(a), "l"(b), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* m, uint32_t bar, int x, int y) {
  asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               ::"r"(dst), "l"((uint64_t)m), "r"(bar), "r"(x), "r"(y) : "memory");
}
__device__ __forceinline__ void tma2d_hint(uint32_t dst, const CUtensorMap* m, uint32_t bar, int x, int y, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
      ::"r"(dst), "l"((uint64_t)m), "r"(bar), "r"(x), "r"(y), "l"(pol) : "memory");
}
#endif

template <int KS>
__global__ void __launch_bounds__(kThreads, 1) tc_k48_kernel(const __grid_constant__ CUtensorMap mapW128,
                                                            const __grid_constant__ CUtensorMap mapW64,
                                                            const __grid_constant__ CUtensorMap mapE128,
                                                            const __grid_constant__ CUtensorMap mapE64,
                                                            const float* __restrict__ bias, OutPtrs outs, int B,
                                                            int Ntile, int nb, int stages) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  extern __shared__ uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t full[kMaxStages], empty[kMaxStages], accum;
  __shared__ uint32_t tmem_slot;
  const uint32_t sbase = (smem_u32(smem_raw) + 1023) & ~1023u;
  const int half = Ntile / 2, N2 = Ntile - 256, h2 = N2 / 2;
  const uint32_t oE128 = kW128, oW64 = kW128 + half * 128, oE64 = oW64 + kW64;
  const uint32_t loaded = KS == 48 ? (uint32_t)(kW128 + kW64 + half * 192) : (uint32_t)(kW128 + half * 128);
  const uint32_t stride = (loaded + 1023) & ~1023u;
  constexpr int nIt = kD / KS;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const int cid = blockIdx.x >> 1, wt = cid / nb, bt = cid - wt * nb;
  const int m0 = wt * 2 * kRows + (int)rank * kRows, n0 = bt * Ntile;
  if (threadIdx.x == 0) {
    for (int i = 0; i < stages; ++i) { mbar_init(smem_u32(&full[i])); mbar_init(smem_u32(&empty[i])); }
    mbar_init(smem_u32(&accum));
    asm volatile("fence.mbarrier_init.release.cluster;");
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapW128) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapE128) : "memory");
    if (KS == 48) {
      asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapW64) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapE64) : "memory");
    }
  }
  if (warp == 3) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(&tmem_slot)), "r"(512));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  cluster_sync();
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t tmem = tmem_slot;

  if (warp == 0 && lane == 0) {   // TMA producer (both CTAs)
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    const int er = n0 + (int)rank * half;
    for (int i = 0; i < nIt; ++i) {
      const int s = i % stages;
      if (i >= stages) mbar_wait(smem_u32(&empty[s]), ((i / stages) - 1) & 1);
      const uint32_t fl = smem_u32(&full[s]), fb = mapa0(fl);   // leader's full barrier collects both CTAs' bytes
      if (rank == 0)
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(fl), "r"(2 * loaded) : "memory");
      const uint32_t a = sbase + s * stride;
      const int k0 = i * KS;
      tma2d_hint(a, &mapW128, fb, k0, m0, pol);
      tma2d(a + oE128, &mapE128, fb, k0, er);
      if (KS == 48) {
        tma2d_hint(a + oW64, &mapW64, fb, k0 + 32, m0, pol);
        tma2d(a + oE64, &mapE64, fb, k0 + 32, er);
      }
    }
  } else if (warp == 1 && lane == 0 && rank == 0) {   // MMA issuer (leader): two UMMAs 256 x {256, N2} x 8 per k-step
    const uint32_t ib = (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)(2 * kRows >> 4) << 24);
    const uint32_t id1 = ib | ((uint32_t)(256 >> 3) << 17), id2 = ib | ((uint32_t)(N2 >> 3) << 17);
    for (int i = 0; i < nIt; ++i) {
      const int s = i % stages;
      mbar_wait(smem_u32(&full[s]), (i / stages) & 1);
      asm volatile("tcgen05.fence::after_thread_sync;");
      const uint32_t a = sbase + s * stride;
      const uint64_t da = sw128_desc(a), db1 = sw128_desc(a + oE128), db2 = sw128_desc(a + oE128 + 128 * 128);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const uint32_t acc = (i | k) != 0;
        mma2(tmem, da + 2 * k, db1 + 2 * k, id1, acc);
        mma2(tmem + 256, da + 2 * k, db2 + 2 * k, id2, acc);
      }
      if (KS == 48) {
        const uint64_t ea = sw64_desc(a + oW64), eb1 = sw64_desc(a + oE64), eb2 = sw64_desc(a + oE64 + 128 * 64);
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          mma2(tmem, ea + 2 * k, eb1 + 2 * k, id1, 1u);
          mma2(tmem + 256, ea + 2 * k, eb2 + 2 * k, id2, 1u);
        }
      }
      umma_commit2(smem_u32(&empty[s]));
    }
    umma_commit2(smem_u32(&accum));
  }
  __syncwarp();

  mbar_wait(smem_u32(&accum), 0);
  asm volatile("tcgen05.fence::after_thread_sync;");
  // 8 warps: warp w reads TMEM lane quadrant w % 4 and every other 16-column chunk.
  const int q = warp & 3, h = warp >> 2;
  const int m = m0 + q * 32 + lane, c = m0 / kD;
  const float bv = bias[m];
  float* oc = outs.p[0];   // select without a dynamic index (which would copy the pointers to local memory)
#pragma unroll
  for (int i = 1; i < 6; ++i)
    if (c == i) oc = outs.p[i];
  float* o = oc + (m - c * kD);
  const uint32_t taddr = tmem + ((uint32_t)(q * 32) << 16);
  for (int cc = h * 16; cc < Ntile; cc += 32) {
    uint32_t r[16];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                 : "r"(taddr + cc));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
    for (int g = 0; g < 2; ++g) {   // region boundaries (128, 256, 256 + h2) are multiples of 8
      const int j = cc + 8 * g;
      const int loc0 = j < 128 ? j : j < 256 ? half + (j - 128) : j - 256 < h2 ? j - 128 : half + (j - 128 - h2);
      float* og = o + (size_t)(n0 + loc0) * kD;
      const int lim = B - (n0 + loc0);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (i < lim) og[(size_t)i * kD] = __uint_as_float(r[8 * g + i]) + bv;
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  cluster_sync();
  if (warp == 3) {
    asm volatile("tcgen05.fence::after_thread_sync;");
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(512));
  }
#endif
}

bool encode(CUtensorMap* mp, const float* p, int rows, int boxk, int boxrows, CUtensorMapSwizzle swz) {
  cuuint64_t d[2] = {(cuuint64_t)kD, (cuuint64_t)rows}, s[1] = {(cuuint64_t)kD * 4};
  cuuint32_t b[2] = {(cuuint32_t)boxk, (cuuint32_t)boxrows}, es[2] = {1, 1};
  return cuTensorMapEncodeTiled(mp, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 2, (void*)p, d, s, b, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <int KS>
int launch(const CUtensorMap* m, const float* bias, const OutPtrs& op, int B, int Ntile, int nb, int stages, int smem,
           cudaStream_t stream) {
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(tc_k48_kernel<KS>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemDyn);
    attr_set = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(2 * kWTiles * nb);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 2;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, tc_k48_kernel<KS>, m[0], m[1], m[2], m[3], bias, op, B, Ntile, nb, stages);
  return (int)cudaGetLastError();
}
}  // namespace

extern "C" int adamod_tc_k48(const float* emb, const float* W, const float* bias, float* const* outs, int B,
                             cudaStream_t stream) {
  if (B <= 256) return 440;
  const int nb = (B + 511) / 512;
  const int Ntile = ((B + nb - 1) / nb + 15) / 16 * 16;   // in (256, 512] for every B > 256
  const int ks = Ntile <= 432 ? 48 : 32;   // 48-k stages where 32-k stages sit on the floor (MMA < ~900 clk)
  const int half = Ntile / 2;
  const int loaded = ks == 48 ? kW128 + kW64 + half * 192 : kW128 + half * 128;
  const int stride = (loaded + 1023) / 1024 * 1024, avail = kSmemDyn - 1024;
  int stages = avail / stride;   // ks 48: 3 at Ntile 272-432; ks 32: 5 at 448-464, 4 at 480-512
  if (stages > kMaxStages) stages = kMaxStages;
  if (stages < 2) return 441;
  CUtensorMap m[4];
  if (!encode(&m[0], W, kN, 32, kRows, CU_TENSOR_MAP_SWIZZLE_128B) || !encode(&m[1], W, kN, 16, kRows, CU_TENSOR_MAP_SWIZZLE_64B) ||
      !encode(&m[2], emb, B, 32, half, CU_TENSOR_MAP_SWIZZLE_128B) || !encode(&m[3], emb, B, 16, half, CU_TENSOR_MAP_SWIZZLE_64B))
    return 442;
  OutPtrs op;
  for (int i = 0; i < 6; ++i) op.p[i] = outs[i];
  const int smem = stages * stride + 1024;
  return ks == 48 ? launch<48>(m, bias, op, B, Ntile, nb, stages, smem, stream)
                  : launch<32>(m, bias, op, B, Ntile, nb, stages, smem, stream);
}
```

```cuda file=kernel.cu
#include <cuda_runtime.h>

// One launch per call; the kernel is chosen from B only.
//   B <= 131: hand-written tcgen05 TF32 kernel, one-SM CTAs, evict_first W stream (tc_small.cu).
//   131 < B <= 256: hand-written 2SM tcgen05 TF32 kernel, cluster of 2, 2-k-block stages (tc_pair.cu).
//   B > 256: hand-written 2SM kernel with two TMEM accumulators, ceil(B / 512) batch tiles of <= 512 columns,
//            48-k stages (SW128 + SW64 boxes) for tiles <= 432 columns, 32-k stages above (tc_k48.cu).
extern "C" int adamod_tc_small(const float*, const float*, const float*, float* const*, int, cudaStream_t);
extern "C" int adamod_tc_pair(const float*, const float*, const float*, float* const*, int, cudaStream_t);
extern "C" int adamod_tc_k48(const float*, const float*, const float*, float* const*, int, cudaStream_t);

extern "C" int adamod_launch(const float* emb, const float* weight, const float* bias, float* const* outs, int B,
                             cudaStream_t stream) {
  if (B <= 131) return adamod_tc_small(emb, weight, bias, outs, B, stream);
  if (B <= 256) return adamod_tc_pair(emb, weight, bias, outs, B, stream);
  return adamod_tc_k48(emb, weight, bias, outs, B, stream);
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

extern "C" int adamod_launch(const float* emb, const float* weight, const float* bias, float* const* outs, int B,
                             cudaStream_t stream);

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
  int st = adamod_launch(emb.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), outs, B, stream);
  TORCH_CHECK(st == 0, "adamod_launch failed with status ", st);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "AdaLayerNormZero modulation: tcgen05 TF32 GEMM with bias and six-way split epilogue");
}
```

```yaml design-card
id: r5-e3-tc-k48-pair
parents:
  - r4-e1-tc-pair2-twoacc
  - r4-e2-tc-pair2-nb2
operation: structural_mutation
language: cuda_cpp
niche:
  mem: tma-tensor
  st: direct
  grid: oneshot
  launch: fused
  tile: rows128
  red: cta
  cache: stream
  spec: dispatch:size
hypothesis: >-
  E3/H26: each K stage is 48 k instead of 32 (an SW128 32-k box plus an SW64 16-k box per operand, 12 UMMAs and one
  commit per stage, a 3-stage ring of 58-66 KB). This puts 1.5x the MMA work behind each commit and hides the
  ~900-1200 clk per-stage floor of 32-k rings wherever N < ~470. One kernel (two TMEM accumulators, nb = ceil(B/512)
  batch tiles, no CUTLASS) covers every B > 256. With L2-hot W, the 48-k stage at Bpad 384 runs at the MMA ideal
  (1175 vs 1152 clk). With real W, a 3-deep ring exposes about 10% DRAM latency (1290-1340 clk). That exposure is set
  in ns, so it should cost fewer cycles at the portal's 1.5 GHz lock. Rented, same GPU vs r3-e2: 373 -7.8%,
  384 -4.7%, 449 -11.3%, 512 -4.0%, 691 -0.9%, 773 -8.8%, 853 -4.1%, 919 +1.3%.
expected_effect:
  S:
    pct: 0
    confidence: high
  M:
    pct: -8
    confidence: low
  L:
    pct: -6
    confidence: low
resources_sm100a:
  regs_per_thread: 38
  smem_per_cta_bytes: 217088
  threads_per_cta: 256
  ctas_per_sm: 1
  bytes_in_flight_per_sm: 122880
  launches_per_call: 1
knobs:
  KS_RULE: "48 if Ntile <= 432 else 32"
  NB_RULE: "ceil(B / 512); Ntile = roundup16(ceil(B / nb))"
  STAGES_RULE: "(226 KB - 1 KB) / roundup1024(stage bytes): ks48 -> 3 (Ntile 272-432), ks32 -> 5 (448-464), 4 (480-512)"
  STAGE_LAYOUT: "[W128 16 KB][E128 half x 128 B][W64 8 KB][E64 half x 64 B]"
  SW64_DESC: "SBO 512 B, layout type 4, start advanced 32 B per K-step"
  EPILOGUE_WARPS: 8
  W_HINT: evict_first
  TMA_DTYPE: TFLOAT32
  CLUSTER: 2
dispatch:
  - max_tokens: 131
    kernel: tc_small_kernel (unchanged r2 path)
    meta:
      launches: 1
  - max_tokens: 256
    kernel: tc_pair_kernel (unchanged r3 path)
    meta:
      launches: 1
  - max_tokens: 432
    kernel: tc_k48_kernel<48>
    meta:
      nb: 1
      stages: 3
      launches: 1
  - max_tokens: 512
    kernel: tc_k48_kernel<32>
    meta:
      nb: 1
      stages: 5-4
      launches: 1
  - max_tokens: 864
    kernel: tc_k48_kernel<48>
    meta:
      nb: 2
      stages: 3
      launches: 1
  - max_tokens: 1024
    kernel: tc_k48_kernel<32>
    meta:
      nb: 2
      stages: 5-4
      launches: 1
  - max_tokens: null
    kernel: tc_k48_kernel<48|32>
    meta:
      nb: ceil(B/512)
      launches: 1
tests:
  - H26
  - H23
  - H22
  - H19
findings:
  - "run_tests 16/16 PASS (10 rounds each), one launch, no CUTLASS. Rented us vs r3-e2-tc-pair-kbs2 on the same GPU: 373 63.0 vs 68.3 (-7.8%), 384 64.8 vs 67.9 (-4.7%), 449 70.6 vs 79.7 (-11.3%), 512 77.0 vs 80.2 (-4.0%), 691 108.7 vs 109.6 (-0.9%), 773 118.5 vs 129.9 (-8.8%), 853 126.1 vs 131.5 (-4.1%), 919 134.2 vs 132.5 (+1.3%); B <= 256 unchanged code within +-1.9%. Band S -0.4% M -2.3% L -4.7%; emulator 0.6291 +- 0.012 (P beat 88%) [run_tests]"
  - "SW64 UMMA descriptor (start>>4, LBO 1, SBO 512 B, version bit 46, layout type 4 at bits 61-63, advanced by 32 B per K=8 step) with a TMA SWIZZLE_64B 16-float box is correct on the first build. Max |err| vs fp64 (W/sqrt(3072), bias randn) is bit-identical to the 32-k ring (same accumulation order): 373 1.56e-3, 384 1.53-1.56e-3, 449 1.51-1.68e-3, 512 1.63e-3, 691 1.59-1.62e-3, 773 1.62e-3, 853 1.59e-3, 919 1.53e-3; 0 elements outside atol+rtol, 4 draws [probe_b200]"
  - "cycles per stage, real W after the 252 MB zero flush, evict_first, clock64 + globaltimer, rented 1.61-1.74 GHz, 3 interleaved rounds, two containers. Bpad 384 nb1: k32 st4/5 893-908 (container A) and 932-936 (B); k48 st3 1307-1341 (A, 871-894 per 32-k) and 1281-1301 (B, 854-868 per 32-k); MMA ideal 1152 per 48-k. So -2.5% (A) and -7.5% (B) per K; span 97.5-100k vs 100-104k clk. Success target (span <= 90k) missed. Refutation threshold (>= 1350 clk) not reached [probe_b200]"
  - "classification at Bpad 384 k48: with L2-hot W (4 W tiles) 1175 clk = 98% of MMA ideal, so the per-stage floor is hidden; with real W 1290-1300, so about 10% is exposed DRAM latency of the 3-deep ring. Feed-only (no MMA) with real W runs 1208 clk at 1.96 GHz = 617 ns per stage = the 40 us evict_first stream floor, against MMA 707 ns at 1.63 GHz: feed and MMA overlap only partly with 2 stages of lead. k32 with hot W stays at 930 (floor-bound whatever W does) [probe_b200]"
  - "k48 at 449/512 nb1 loses: 449 1430-1510 per 48-k (953-1007 per 32-k) vs k32 st5 941-947; 512 1569-1587 vs 1037-1044. Both are MMA-bound already at k32, and 3 stages expose latency (hot W 449 k48 1418 = k32). Shipped: k32 at Ntile > 432 [probe_b200]"
  - "L band nb2 per 48-k stage (per 32-k) vs k32 st5: Ntile 352 (691) 1145-1166 (763-777) vs 891 (-13%, span 86-87k vs 98k); Ntile 400 (773) 1234-1250 (823-834) vs 892 (-7.5%); 432 (853) 1318-1331 (879-888) vs 894 (-1%); 464 (919) 1418 vs 940 (0%). nb2 tiles are not DRAM-bound (sibling pair re-reads W from L2), so k48 reaches 91-97% of MMA ideal [probe_b200]"
  - "2-stage k48 ring at Bpad 384: 1736-1799 clk per stage (+35%). The 3-stage requirement holds [probe_b200]"
  - "dead end: decoupled rings (separate W and emb rings, separate full/empty barriers and producer threads, two commits per stage) cost +390-430 clk per stage at every shape. k32 1322 vs 896-947 at 384/449/512/919; k48 per 32-k 1023-1038 vs 763-935 at 384/691/773/853; deeper W rings (4-6) change nothing. The per-stage floor scales with the number of tcgen05.commit per stage, not ring depth [probe_b200]"
  - "dead end: L2 prefetch of W ahead of the ring (cp.async.bulk.prefetch.tensor.2d.L2, distance 3/4/6 stages, unhinted or evict_first-hinted) is +18-23% at Bpad 384 k48 (1518-1643 vs 1281-1324 clk) and +3-4% at k32 384/449 [probe_b200]"
  - "dead end: L2 promotion NONE/128B/256B on the W maps changes nothing (k48 384: 1315-1326 vs 1290-1301; k32 932-940). W evict_normal instead of evict_first: k48 384 +15% (1468-1484), k32 +3% [probe_b200]"
  - "epilogue: indexing the OutPtrs kernel parameter with the runtime chunk c made ptxas create a 48-B stack frame (the 6 pointers copied to local memory). A select loop over the 6 pointers removes it (0 B stack, 38 regs). The epilogue costs 8.4-11.3k clk per CTA at 373-512 and 8.1-10.1k at nb2, about 7-10% of the span; it is bound by store bandwidth (about 5-6 TB/s) [probe_b200, compile_b200]"
  - "static sm_100a: tc_k48_kernel<48> and <32> 38 regs, 0 local, UTMALDG x6, STG x32; dynamic smem 181 KB (ks48 Bpad 384), 205 KB (ks48 Ntile 432), 201-212 KB (ks32) [compile_b200]"
  - "portal prediction: rented cycle gains are -5..-8% at 373/384 (DRAM-latency-exposed, ns-bound), -11% at 449, -7..-9% at 773 over CUTLASS. At the 1.5 GHz lock the 48-k ring's DRAM exposure costs fewer cycles (617 ns feed vs 768 ns MMA per stage), so 373/384 should approach 64 x ~1190 + ~13k = ~89k clk = ~60-66 us vs portal CUTLASS 83 (-20..-28%). By the operator rule max(1.05 x DRAM, cycles/1.55 GHz) + 11% it is about 70 us (-15%). 691/919 are about even with CUTLASS rented; the r3 calibration (hand-written kernels gained more on the portal than rented) favours keeping them hand-written [analysis]"
  - "score model: M -8% / L -6% vs r3-e2 gives 0.6333 (current 0.6166); a 10% cut is worth +0.0105 in M and +0.0145 in L [predict_score]"
paths:
  - max_tokens: 131
    lang: cuda
    width: 128
    threads: 128
    rows: 128
    grid: oneshot
    mem: tma
    launches: 1
    cluster: 1
    x: none
    w: ef
    st: none
  - max_tokens: 256
    lang: cuda
    width: 128
    threads: 128
    rows: 256
    grid: oneshot
    mem: tma
    launches: 1
    cluster: 2
    x: none
    w: ef
    st: none
  - max_tokens: 432
    lang: cuda
    width: 128
    threads: 256
    rows: 256
    grid: oneshot
    mem: tma
    launches: 1
    cluster: 2
    x: none
    w: ef
    st: none
  - max_tokens: 512
    lang: cuda
    width: 128
    threads: 256
    rows: 256
    grid: oneshot
    mem: tma
    launches: 1
    cluster: 2
    x: none
    w: ef
    st: none
  - max_tokens: 864
    lang: cuda
    width: 128
    threads: 256
    rows: 256
    grid: oneshot
    mem: tma
    launches: 1
    cluster: 2
    x: none
    w: ef
    st: none
  - max_tokens: null
    lang: cuda
    width: 128
    threads: 256
    rows: 256
    grid: oneshot
    mem: tma
    launches: 1
    cluster: 2
    x: none
    w: ef
    st: none
runs_on:
  H200:
    runs: false
    representative: false
    note: "tcgen05/TMEM/cta_group::2 are sm_100a only"
  A100:
    runs: false
    representative: false
  L40S:
    runs: false
    representative: false
risks:
  - "The 373/384 gain on the rented card is only -5..-8% because the 3-deep 48-k ring exposes DRAM latency. The larger portal gain rests on that exposure being ns-bound (fewer cycles at 1.5 GHz); if the portal DRAM sits further away in cycles, 373/384 land near -5%"
  - "691 and 919 move from CUTLASS to the hand-written kernel at roughly rented parity (-0.9% / +1.3%); if hand-written kernels scale worse to the portal than CUTLASS (ratio 1.31 vs 1.25-1.27), these two could lose 2-4%"
  - "Two waves of 74 pairs at nb2 assume 74 co-resident 2-CTA clusters at up to 217 KB smem (true on the rented card)"
  - "kSmemDyn 226 KB leaves under 1 KB of headroom against the 227 KB per-CTA limit at ks32 Ntile 448-464 (5 x 45 KB + 1 KB)"
  - "mbar_wait traps after 2^27 polls, so a broken pipeline errors instead of hanging"
measure_first:
  - B200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```