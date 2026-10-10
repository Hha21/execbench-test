### Rationale
The new kernel passes all 16 workloads but misses the success bar, so H22 is refuted as stated. I built it from scratch because no E1 kernel was in the archive. It is a 2-CTA pair with two TMEM accumulators, two batch tiles per W tile and 144 clusters, and it is shipped for 720 < B ≤ 1024. Against CUTLASS 256x192 in the same container it is 6.5% faster at 773, 6.8% at 853, 1% at 919 and 8% slower at 691. The target was at least 15% fewer cycles, and 773/853 land inside the "within 8%" refutation band. The cause is not the ~75% ingest ceiling: every 1-k-block stage costs about 960–985 clocks whatever its MMA work or byte count. Two side results matter more:
- **Epilogue bug:** remapping each column separately pushed `r[]` into local memory and cost 30k clocks per CTA. Remapping once per 8 columns fixes it.
- **Single-tile variant:** the same kernel with nb = 1 beats the best CUTLASS tile by 8–15% at B = 373–512 (E1's target band). It is not shipped, because the task said to leave that path alone.

```json solution-spec
{"name": "r4-e2-tc-pair2-nb2", "definition": "035_flux_ada_layer_norm_zero_modulation_extraction", "author": "solx-loop", "description": "2SM two-accumulator tcgen05 TF32 pair kernel, two batch tiles per W tile for 720 < B <= 1024; other paths as r3-e2-tc-pair-kbs2", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": [], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17", "--expt-relaxed-constexpr"]}}}
```

```cuda file=adamod_gemm.cuh
#pragma once
// TF32 tcgen05 GEMM for AdaLayerNormZero modulation: D[m = feature, n = batch] = W[m, :] . emb[n, :] + bias[m],
// built from CUTLASS 4.4 sm100 collectives (2SM MMA, cluster 2x1, persistent CLC tile scheduler).
// A = W [18432, 3072] K-major, B = emb [B, 3072] K-major: the large dimension is M, as cuBLAS's tnn kernels do,
// so a 256-row W tile feeds every batch tile of the same wave and W is read from DRAM once.
// Epilogue: CUTLASS's no-smem (direct store) epilogue with C = bias broadcast (stride 0 along the batch mode),
// wrapped so that each tile writes straight into its own output tensor: 3072 % tile_m == 0, so the tile's
// chunk c = m0 / 3072 is uniform and D is re-based to outs[c] - c * 3072 (column-major, ld 3072).
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace adamod {
using namespace cute;
constexpr int kD = 3072;
constexpr int kN = 6 * kD;

// The base epilogue keeps a const reference to its Params; we own that Params object and re-point ptr_D per tile.
template <class Base>
struct HeldParams {
  typename Base::Params held;
};

template <class Base>
class ChunkedEpilogue : private HeldParams<Base>, public Base {
  using Holder = HeldParams<Base>;

 public:
  struct Arguments {
    typename Base::Arguments base;
    float* outs[6];
  };
  struct Params {
    typename Base::Params base;
    float* outs[6];
  };

  template <class PS>
  static constexpr Params to_underlying_arguments(PS const& ps, Arguments const& a, void* ws) {
    Params p{Base::to_underlying_arguments(ps, a.base, ws), {}};
    for (int i = 0; i < 6; ++i) p.outs[i] = a.outs[i];
    return p;
  }
  template <class PS>
  static size_t get_workspace_size(PS const& ps, Arguments const& a) {
    return Base::get_workspace_size(ps, a.base);
  }
  template <class PS>
  static cutlass::Status initialize_workspace(PS const& ps, Arguments const& a, void* ws, cudaStream_t s,
                                              cutlass::CudaHostAdapter* h = nullptr) {
    return Base::initialize_workspace(ps, a.base, ws, s, h);
  }
  template <class PS>
  static bool can_implement(PS const& ps, Arguments const& a) {
    return Base::can_implement(ps, a.base);
  }
  CUTLASS_DEVICE static void prefetch_tma_descriptors(Params const&) {}

  CUTLASS_DEVICE ChunkedEpilogue(Params const& p, typename Base::TensorStorage& st)
      : Holder{p.base}, Base(Holder::held, st), outs_(p.outs) {}

  template <bool ReuseTmem = false, class LP, class LPS, class SP, class SPS, class AP, class APS, class PS, class CS,
            class CC, class TS, class TM, class Acc, class ST>
  CUTLASS_DEVICE auto store(LP lp, LPS lps, SP sp, SPS sps, AP ap, APS aps, PS ps, CS cs, CC cc, TS ts, TM tm, Acc acc,
                            ST& st) {
    const int c = (int(get<0>(cc)) * int(size<0>(cs))) / kD;   // chunk of this CTA's rows
    Holder::held.ptr_D = outs_[c] - (size_t)c * kD;             // feature m -> outs[c][b * 3072 + m - c * 3072]
    return Base::template store<ReuseTmem>(lp, lps, sp, sps, ap, aps, ps, cs, cc, ts, tm, acc, st);
  }

 private:
  float* const* outs_;
};

// TM x TN MMA tile over a 2-CTA pair; Stages = 0 lets CUTLASS fill shared memory.
template <int TM, int TN, int Stages>
struct Tf32Gemm {
  using MmaTile = Shape<Int<TM>, Int<TN>, _32>;
  using Cluster = Shape<_2, _1, _1>;
  using EpiBase = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, float, cutlass::layout::ColumnMajor, 4, float,
      cutlass::layout::ColumnMajor, 4, cutlass::epilogue::NoSmemWarpSpecialized2Sm,
      cutlass::epilogue::fusion::LinearCombination<float, float, float, float>>::CollectiveOp;
  using Epi = ChunkedEpilogue<EpiBase>;
  using StageSpec = cute::conditional_t<(Stages > 0), cutlass::gemm::collective::StageCount<Stages>,
                                        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                                            sizeof(typename EpiBase::SharedStorage))>>;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, cutlass::tfloat32_t, cutlass::layout::RowMajor, 4,
      cutlass::tfloat32_t, cutlass::layout::ColumnMajor, 4, float, MmaTile, Cluster, StageSpec,
      cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epi, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// fp32 data is read by TMA as tf32; accumulation is fp32 over K.
template <class G>
int run_gemm(const float* emb, const float* W, const float* bias, float* const* outs, int B, int device, int sms,
             cudaStream_t stream) {
  using Kernel = typename G::Kernel;
  using Gemm = typename G::Gemm;
  using SA = typename Kernel::StrideA;
  using SB = typename Kernel::StrideB;
  using SC = typename G::EpiBase::StrideC;
  using SD = typename G::EpiBase::StrideD;
  typename Gemm::Arguments args;
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = {kN, B, kD, 1};
  args.mainloop.ptr_A = reinterpret_cast<const cutlass::tfloat32_t*>(W);
  args.mainloop.dA = cutlass::make_cute_packed_stride(SA{}, make_shape(kN, kD, 1));
  args.mainloop.ptr_B = reinterpret_cast<const cutlass::tfloat32_t*>(emb);
  args.mainloop.dB = cutlass::make_cute_packed_stride(SB{}, make_shape(B, kD, 1));
  args.epilogue.base.thread.alpha = 1.f;
  args.epilogue.base.thread.beta = 1.f;
  args.epilogue.base.ptr_C = bias;   // C[m, n] = bias[m]
  SC dC{};
  get<1>(dC) = 0;
  get<2>(dC) = 0;
  args.epilogue.base.dC = dC;
  args.epilogue.base.ptr_D = outs[0];   // replaced per tile by ChunkedEpilogue
  SD dD{};
  get<1>(dD) = kD;
  get<2>(dD) = 0;
  args.epilogue.base.dD = dD;
  for (int i = 0; i < 6; ++i) args.epilogue.outs[i] = outs[i];
  args.hw_info.device_id = device;
  args.hw_info.sm_count = sms;
  Gemm gemm;
  if (Gemm::get_workspace_size(args) != 0) return 900;   // no workspace (and no memset) by construction
  cutlass::Status s = gemm.can_implement(args);
  if (s != cutlass::Status::kSuccess) return 100 + (int)s;
  s = gemm.initialize(args, nullptr, stream);
  if (s != cutlass::Status::kSuccess) return 200 + (int)s;
  s = gemm.run(stream);
  if (s != cutlass::Status::kSuccess) return 300 + (int)s;
  return (int)cudaGetLastError();
}
}  // namespace adamod
```

```cuda file=tile_256x192.cu
#include "adamod_gemm.cuh"

extern "C" int adamod_gemm_256x192(const float* emb, const float* W, const float* bias, float* const* outs, int B,
                                   int device, int sms, cudaStream_t stream) {
  return adamod::run_gemm<adamod::Tf32Gemm<256, 192, 0>>(emb, W, bias, outs, B, device, sms, stream);
}
```

```cuda file=tile_256x256.cu
#include "adamod_gemm.cuh"

extern "C" int adamod_gemm_256x256(const float* emb, const float* W, const float* bias, float* const* outs, int B,
                                   int device, int sms, cudaStream_t stream) {
  return adamod::run_gemm<adamod::Tf32Gemm<256, 256, 0>>(emb, W, bias, outs, B, device, sms, stream);
}
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

```cuda file=tc_pair2.cu
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>

// 720 < B <= 1024: 2SM tcgen05 TF32 GEMM with two TMEM accumulators and nb = 2 batch tiles per 256-row W tile.
// Ntile = roundup(ceil(B / nb), 16) in (256, 512]: one pass over K computes N1 = 256 columns into TMEM columns
// [0, 256) and N2 = Ntile - 256 columns into [256, Ntile), both from the same A (W) stage, 8 UMMAs per 1-k-block stage.
// Each CTA of the pair loads Ntile / 2 contiguous emb rows (n0 + rank * Ntile / 2 ...); the UMMA B operand of N columns
// takes N / 2 rows from each CTA, so accumulator columns map to emb rows as
//   j < 128 -> j, 128..255 -> half + j - 128, 256..256+h2 -> j - 128, rest -> half + j - 128 - h2.
// Grid: 72 W tiles x nb batch tiles = 144 clusters of 2 (two waves of 74 pairs); cluster id -> (W tile = id / nb,
// batch tile = id % nb), so the two batch tiles of a W tile run together and the second read of W hits L2.
// W loads carry L2 evict_first; TMA maps are TFLOAT32 (round on load); emb rows >= B are zero-filled by TMA and the
// epilogue skips them. The epilogue remaps once per 8 columns: a per-column remap put r[] in local memory and cost
// 30k clk per CTA (probe).
namespace {
constexpr int kD = 3072, kN = 6 * kD, kRows = 128, kBK = 32, kKB = kD / kBK, kWBlk = kRows * kBK * 4;
constexpr int kMaxStages = 8, kSmemDyn = 226 * 1024, kWTiles = kN / (2 * kRows);
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

__global__ void __launch_bounds__(128, 1) tc_pair2_kernel(const __grid_constant__ CUtensorMap mapW,
                                                         const __grid_constant__ CUtensorMap mapE,
                                                         const float* __restrict__ bias, OutPtrs outs, int B, int Ntile,
                                                         int nb, int stages) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  extern __shared__ uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t full[kMaxStages], empty[kMaxStages], accum;
  __shared__ uint32_t tmem_slot;
  const uint32_t sbase = (smem_u32(smem_raw) + 1023) & ~1023u;
  const int half = Ntile / 2, N2 = Ntile - 256, h2 = N2 / 2;
  const uint32_t eBlk = half * 128, stageBytes = kWBlk + eBlk;   // per CTA
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const int cid = blockIdx.x >> 1, wt = cid / nb, bt = cid - wt * nb;
  const int m0 = wt * 2 * kRows + (int)rank * kRows, n0 = bt * Ntile;
  if (threadIdx.x == 0) {
    for (int i = 0; i < stages; ++i) { mbar_init(smem_u32(&full[i])); mbar_init(smem_u32(&empty[i])); }
    mbar_init(smem_u32(&accum));
    asm volatile("fence.mbarrier_init.release.cluster;");
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapE) : "memory");
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
    for (int i = 0; i < kKB; ++i) {
      const int s = i % stages;
      if (i >= stages) mbar_wait(smem_u32(&empty[s]), ((i / stages) - 1) & 1);
      const uint32_t fl = smem_u32(&full[s]), fb = mapa0(fl);   // leader's full barrier collects both CTAs' bytes
      if (rank == 0)
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(fl), "r"(2 * stageBytes) : "memory");
      const uint32_t a = sbase + s * stageBytes, b = a + kWBlk;
      asm volatile(
          "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5}], [%2], %6;"
          ::"r"(a), "l"((uint64_t)&mapW), "r"(fb), "r"(0), "r"(m0), "r"(i), "l"(pol) : "memory");
      asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
                   ::"r"(b), "l"((uint64_t)&mapE), "r"(fb), "r"(i * kBK), "r"(n0 + (int)rank * half) : "memory");
    }
  } else if (warp == 1 && lane == 0 && rank == 0) {   // MMA issuer (leader): two UMMAs 256 x {256, N2} x 8 per k-step
    const uint32_t ib = (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)(2 * kRows >> 4) << 24);
    const uint32_t id1 = ib | ((uint32_t)(256 >> 3) << 17), id2 = ib | ((uint32_t)(N2 >> 3) << 17);
    for (int i = 0; i < kKB; ++i) {
      const int s = i % stages;
      mbar_wait(smem_u32(&full[s]), (i / stages) & 1);
      asm volatile("tcgen05.fence::after_thread_sync;");
      const uint32_t a = sbase + s * stageBytes, b = a + kWBlk;
      const uint64_t da = sw128_desc(a), db = sw128_desc(b), db2 = sw128_desc(b + 128 * 128);
#pragma unroll
      for (int k = 0; k < kBK / 8; ++k) {
        const uint32_t acc = (i | k) != 0;
        asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, p; }"
                     ::"r"(tmem), "l"(da + 2 * k), "l"(db + 2 * k), "r"(id1), "r"(acc));
        asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, p; }"
                     ::"r"(tmem + 256), "l"(da + 2 * k), "l"(db2 + 2 * k), "r"(id2), "r"(acc));
      }
      umma_commit2(smem_u32(&empty[s]));
    }
    umma_commit2(smem_u32(&accum));
  }
  __syncwarp();

  mbar_wait(smem_u32(&accum), 0);
  asm volatile("tcgen05.fence::after_thread_sync;");
  const int m = m0 + warp * 32 + lane, c = m0 / kD;
  const float bv = bias[m];
  float* o = outs.p[c] + (m - c * kD);
  const uint32_t taddr = tmem + ((uint32_t)(warp * 32) << 16);
  for (int cc = 0; cc < Ntile; cc += 16) {
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
}  // namespace

extern "C" int adamod_tc_pair2(const float* emb, const float* W, const float* bias, float* const* outs, int B, int nb,
                               cudaStream_t stream) {
  const int Ntile = ((B + nb - 1) / nb + 15) / 16 * 16;
  if (B < 1 || Ntile <= 256 || Ntile > 512) return 420;
  const int half = Ntile / 2;
  const int perK = kWBlk + half * 128, avail = kSmemDyn - 1024;
  int stages = avail / perK;   // 5 at Ntile 352-464
  if (stages > kMaxStages) stages = kMaxStages;
  CUtensorMap mW, mE;
  cuuint64_t dW[3] = {(cuuint64_t)kBK, (cuuint64_t)kN, (cuuint64_t)kKB}, sW[2] = {(cuuint64_t)kD * 4, (cuuint64_t)kBK * 4};
  cuuint32_t bW[3] = {kBK, kRows, 1}, es[3] = {1, 1, 1};
  if (cuTensorMapEncodeTiled(&mW, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 3, (void*)W, dW, sW, bW, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return 421;
  cuuint64_t dE[2] = {(cuuint64_t)kD, (cuuint64_t)B}, sE[1] = {(cuuint64_t)kD * 4};
  cuuint32_t bE[2] = {kBK, (cuuint32_t)half};
  if (cuTensorMapEncodeTiled(&mE, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 2, (void*)emb, dE, sE, bE, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return 422;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(tc_pair2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemDyn);
    attr_set = true;
  }
  OutPtrs op;
  for (int i = 0; i < 6; ++i) op.p[i] = outs[i];
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(2 * kWTiles * nb);
  cfg.blockDim = dim3(128);
  cfg.dynamicSmemBytes = stages * perK + 1024;
  cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 2;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, tc_pair2_kernel, mW, mE, bias, op, B, Ntile, nb, stages);
  return (int)cudaGetLastError();
}
```

```cuda file=kernel.cu
#include <cuda_runtime.h>

// One launch per call; the kernel is chosen from B only.
//   B <= 131: hand-written tcgen05 TF32 kernel, one-SM CTAs, evict_first W stream (tc_small.cu).
//   131 < B <= 256: hand-written 2SM tcgen05 TF32 kernel, cluster of 2, 2-k-block stages (tc_pair.cu).
//   256 < B <= 720: CUTLASS sm100 2SM TF32 GEMM with fused bias + six-way epilogue (r1 tiles).
//   720 < B <= 1024: hand-written 2SM two-accumulator kernel, 2 batch tiles of <= 512 columns (tc_pair2.cu).
//   B > 1024: CUTLASS 256x192.
extern "C" int adamod_tc_small(const float*, const float*, const float*, float* const*, int, cudaStream_t);
extern "C" int adamod_tc_pair(const float*, const float*, const float*, float* const*, int, cudaStream_t);
extern "C" int adamod_tc_pair2(const float*, const float*, const float*, float* const*, int, int, cudaStream_t);
extern "C" int adamod_gemm_256x192(const float*, const float*, const float*, float* const*, int, int, int, cudaStream_t);
extern "C" int adamod_gemm_256x256(const float*, const float*, const float*, float* const*, int, int, int, cudaStream_t);

extern "C" int adamod_launch(const float* emb, const float* weight, const float* bias, float* const* outs, int B,
                             cudaStream_t stream) {
  if (B <= 131) return adamod_tc_small(emb, weight, bias, outs, B, stream);
  if (B <= 256) return adamod_tc_pair(emb, weight, bias, outs, B, stream);
  if (B > 720 && B <= 1024) return adamod_tc_pair2(emb, weight, bias, outs, B, 2, stream);
  int device = 0;
  cudaGetDevice(&device);
  static int sms = 0;   // device property, queried once
  if (sms == 0) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (B <= 384) return adamod_gemm_256x192(emb, weight, bias, outs, B, device, sms, stream);
  if (B <= 512) return adamod_gemm_256x256(emb, weight, bias, outs, B, device, sms, stream);
  return adamod_gemm_256x192(emb, weight, bias, outs, B, device, sms, stream);
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
id: r4-e2-tc-pair2-nb2
parents:
  - r3-e2-tc-pair-kbs2
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
  H22 test. The kernel is a 2-CTA pair (cta_group::2, M = 256) with two TMEM accumulators (N1 = 256, N2 = Ntile - 256)
  and nb = 2 batch tiles of Ntile = roundup(ceil(B/2), 16) per W tile. The grid is 144 clusters ordered W-tile-adjacent,
  with kbs = 1, 5 stages and evict_first on W. It covers 720 < B <= 1024 in one launch. It does NOT reach 85% per clock.
  The main loop is floored at 960-985 clk per 1-k-block stage regardless of MMA work or stage bytes, and the two waves
  cannot overlap the epilogue, so it beats CUTLASS 256x192 by only 6.5-6.8% at 773/853 and 1% at 919, and loses 8% at
  691. B = 691 therefore stays on CUTLASS. H22 is refuted as stated (needed >= 15%), but the cause is a per-stage floor
  (H23), not the ingest ceiling (H24).
expected_effect:
  S:
    pct: 0
    confidence: high
  M:
    pct: 0
    confidence: high
  L:
    pct: -3
    confidence: low
resources_sm100a:
  regs_per_thread: 64
  smem_per_cta_bytes: 231424
  threads_per_cta: 128
  ctas_per_sm: 1
  bytes_in_flight_per_sm: 184320
  launches_per_call: 1
knobs:
  NB: 2
  KBS: 1
  STAGES_RULE: "(226 KB - 1 KB) / (16 KB + Ntile/2 * 128 B): 5 at Ntile 352-464"
  CLUSTER_ORDER: "W tile = id / nb, batch tile = id % nb"
  W_HINT: evict_first
  TMA_DTYPE: TFLOAT32
  PAIR2_RANGE: "720 < B <= 1024"
dispatch:
  - max_tokens: 131
    kernel: tc_small_kernel
    meta:
      launches: 1
  - max_tokens: 256
    kernel: tc_pair_kernel
    meta:
      launches: 1
  - max_tokens: 384
    kernel: Tf32Gemm<256,192,0>
    meta:
      launches: 1
  - max_tokens: 512
    kernel: Tf32Gemm<256,256,0>
    meta:
      launches: 1
  - max_tokens: 720
    kernel: Tf32Gemm<256,192,0>
    meta:
      launches: 1
  - max_tokens: 1024
    kernel: tc_pair2_kernel
    meta:
      nb: 2
      launches: 1
  - max_tokens: null
    kernel: Tf32Gemm<256,192,0>
    meta:
      launches: 1
tests:
  - H22
  - H23
  - H24
  - H19
findings:
  - "VERDICT H22: refuted as stated. Same-container harness_time, 3 interleaved rounds, us pair2 (fixed epilogue) vs CUTLASS 256x192 / 256x256: 691 117.0 vs 108.7 / 108.7 (+8%); 773 120.2 vs 128.4 / 133.4 (-6.4%); 853 121.2 vs 129.9 / 135.8 (-6.7%); 919 129.5 vs 130.8 / 136.9 (-1.0%). Needed >= 15% at 773-919 and >= 10% at 691 [probe_b200]"
  - "run_tests 16/16 PASS, one launch. us vs r2-e1-tc05-ef-small on the same GPU: 773 123.9 vs 130.7 (-5.2%), 853 125.5 vs 133.1 (-5.7%), 919 133.2 vs 134.2 (-0.8%); every other B is the r3 code (192 -6.2%, 211 -3.6%, rest within +-1.8%); band S -0.2% M -1.6% L -1.2%; emulator 0.6164 +- 0.012 (P beat 69%) [run_tests]"
  - "main-loop floor: in-kernel clock64/globaltimer, harness flush, real W, rented clock 1.63-1.78 GHz. Main loop per 1-k-block stage with 8 UMMAs: 958-985 clk at Ntile 352/400/432/464 and also at nb=3 (Ntile 320). MMA ideal per stage is 2*Ntile = 640-928 clk, so it is MMA-bound only at 464 (94%) [probe_b200]"
  - "floor ablation (B=919 nb2 / 691 nb2 / 919 nb3, clk per k-block): base 985/977/972; second accumulator removed (MMA work halved) 982/981/974; no MMA (TMA + barriers only) 646/641/619 at 1.96 GHz; warm W (4 W tiles) 967/958/957; warm W + no MMA 607/605/601; test_wait spin instead of try_wait 985/980/970. The ~970 floor depends on neither MMA work, stage bytes (33-46 KB), DRAM, nor the wait instruction: H23's max() model fails (floor ~960, not 551-674, at 8 UMMAs per stage with rotating stages) [probe_b200]"
  - "H24 not the limiter here: stage bytes 36 KB (nb3) and 46 KB (nb2 at 919) give the same ~970 clk per stage, so ingest bytes/clk does not set the pace of this ring. CUTLASS 256x192 runs about 460-550 clk per 4-UMMA k-block stage on the same shape (134 us at 919 over ~4.9 tiles per pair), so the hand-written ring's per-stage chain costs about 2x CUTLASS's; cause unresolved [probe_b200, analysis]"
  - "epilogue bug (also a lesson for all hand-written kernels): a per-column piecewise remap of TMEM column -> emb row made ptxas place r[16] in local memory (48 B stack). The epilogue then took 22-33k clk per CTA (1050 clk per 16 columns) even with stores disabled. TMEM loads alone take 1.4k clk for 464 columns; batching 4-8 x16 loads per wait changed nothing. Remapping once per 8-column group (boundaries 128/256/256+h2 are multiples of 8) brings it to ~8k clk with stores: 919 155 -> 129.5 us, 691 139 -> 117 us [probe_b200]"
  - "cluster order: W-tile-adjacent (a) vs batch-major (b, the sibling reads W from DRAM again): 691 139.2 vs 141.2, 773 145.8 vs 147.2, 853 148.7 vs 150.0, 919 154.6 vs 156.4 us (pre-fix epilogue): (a) wins by about 1% only; this kernel is not DRAM-bound, so the second W read costs about 2 us [probe_b200]"
  - "W hint: evict_first vs evict_normal on W within 1% at 691-919 (139.2 vs 139.1, 154.6 vs 154.1) [probe_b200]"
  - "nb=3 at 773/853/919 (Ntile 272/288/320, 3 waves): 193.9/196.7/201.4 us vs nb=2 145-155: +30%. The stage floor is paid per wave, so more tiles only add waves [probe_b200]"
  - "M-band side result for E1/H19: the same kernel with nb=1 (one pass, one wave of 72 pairs) at 373/384/449/512 gives 62.3/62.4/67.1/72.6 us vs the best CUTLASS tile 67.9 (192)/67.9 (192)/79.0 (256)/79.6 (256): -8/-8/-15/-9% in one container, max err 1.53-1.73e-3. Not shipped (task: keep other paths); a strong candidate for the next M-band round [probe_b200]"
  - "precision: max |err| vs fp64 (W/sqrt(3072), bias randn) 691 1.54-1.66e-3, 773 1.63e-3, 853 1.53e-3, 919 1.54-1.64e-3, nb3 919 1.67e-3, nb1 384 1.60-1.73e-3, 449 1.53-1.55e-3, 512 1.59-1.60e-3; one draw at 384 reached 1.73e-3 (above the 1.7e-3 bar, below atol 2e-3) [probe_b200]"
  - "per-clock efficiency (MMA-ideal 2 passes x 96 x 2*Ntile clk / span cycles at ~1.7 GHz): fixed kernel 691 ~68%, 773 ~75%, 853 ~81%, 919 ~81%; CUTLASS 256x192 on its padded work about 77-83% at the same clock. Success needed >= 85% [probe_b200, analysis]"
  - "portal prediction (r3 rule cycles/1.55 GHz): 773/853/919 about 205/207/220k clk -> 132/134/142 us vs CUTLASS portal 164.9/166.7/167.9 (-15..-20%); with the rule's observed +11% underprediction, 147/149/158 us (-6..-11%); with a flat 1.31 clock ratio against CUTLASS's 1.25-1.27, -2/-2/+4%. Central estimate L -3% on the band (3 of 6 workloads) [analysis]"
  - "dead end: kbs=2 does not fit more than 2 stages at Ntile 352-464 (77-91 KB per stage per CTA), and r3 showed 2-stage rings are latency-bound; nb=3 adds a wave [analysis]"
  - "infra: /tmp in probe containers persisted between 2-3 consecutive probes and was then lost (CUTLASS .so had to be rebuilt; 18-28 s for 2 TUs in parallel). Rebuild everything inside the timing probe [probe_b200]"
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
  - max_tokens: 384
    lang: cuda
    width: 128
    threads: 192
    rows: 128
    grid: persistent
    mem: tma
    launches: 1
    cluster: 2
    x: none
    w: none
    st: none
  - max_tokens: 512
    lang: cuda
    width: 128
    threads: 192
    rows: 128
    grid: persistent
    mem: tma
    launches: 1
    cluster: 2
    x: none
    w: none
    st: none
  - max_tokens: 720
    lang: cuda
    width: 128
    threads: 192
    rows: 128
    grid: persistent
    mem: tma
    launches: 1
    cluster: 2
    x: none
    w: none
    st: none
  - max_tokens: 1024
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
  - max_tokens: null
    lang: cuda
    width: 128
    threads: 192
    rows: 128
    grid: persistent
    mem: tma
    launches: 1
    cluster: 2
    x: none
    w: none
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
  - "Rented gain is only 5-6% at 773/853 and about 1% at 919; if the hand-written ring's portal/rented ratio is the clock ratio 1.31 while CUTLASS's stays 1.25-1.27, 919 could lose 3-4% on the portal"
  - "Two waves of 74 clusters at 226 KB smem rely on 74 pairs fitting per wave (true on the rented card); fewer resident pairs on the portal card would add a third partial wave"
  - "kSmemDyn = 226 KB leaves under 1 KB of headroom against the 227 KB per-CTA limit (static smem ~150 B)"
  - "One probe draw at B=384 (nb=1, not shipped) reached max err 1.73e-3; shipped sizes stayed <= 1.67e-3"
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