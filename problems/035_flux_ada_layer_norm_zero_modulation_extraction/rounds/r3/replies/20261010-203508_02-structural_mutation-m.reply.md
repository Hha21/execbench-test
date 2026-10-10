### Rationale
H20 is refuted. I tested the multicast variant against the unchanged tc_small, interleaved in one probe with harness flush and CUPTI, 3 rounds, at an SM clock of 1.66–1.78 GHz. It does not cut cycles anywhere:

| B | baseline cycles | multicast cycles | change |
|---|---|---|---|
| 132 | 74.3–76.2k | 74.9–75.3k | ±1% |
| 192 | 80.1–80.8k | 82.0–83.2k | +2–4% |
| 211 | 80.9–81.4k | 82.8–83.5k | +2–3% |
| 256 | 85.8–87.6k | 86.6–87.7k | 0–1.6% |

A follow-up probe found the real M-band limiter. With W hot in L2, B=192–256 still take 81–82k cycles (86–87k cold), so DRAM is not the limit. Feeding operands with no MMA at all takes 62–64k cycles. That is a per-SM ingest ceiling of about 64–75 B/clk/SM, set by the bytes landing in each SM (W plus the full emb). Multicast halves the L2 reads but not those landing bytes. A cta_group::2 pair (H19/E1) does halve them, so E1 is the route; it does not need emb multicast.

This candidate is the tested multicast build: 16/16 correct and neutral to slightly slower. Do not spend a portal slot on it.

```json solution-spec
{"name": "r3-e3-tc05-emb-mcast2", "definition": "035_flux_ada_layer_norm_zero_modulation_extraction", "author": "solx-loop", "description": "tc_small with TMA multicast of the emb k-blocks across 2-CTA clusters for 131 < B <= 256 (E3/H20); neutral, H20 refuted", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": [], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
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
// kMC (131 < B <= 256): CTAs launch as clusters of 2 that keep their own W tiles; CTA rank r loads emb rows
// [r * Bpad/2, (r+1) * Bpad/2) of every k-block and multicasts them to the same offset in both CTAs (Bpad/2 is a
// multiple of 8, so the halves start on 1024 B swizzle atoms). A stage is refilled only after both CTAs' MMAs have
// read it: the empty barriers take one multicast tcgen05.commit arrival from each CTA.
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
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(cnt));
}
__device__ __forceinline__ void umma_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory");
}
// One arrival on the barrier at the same offset in every CTA of the mask.
__device__ __forceinline__ void umma_commit_mc(uint32_t bar, uint16_t mask) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
               ::"r"(bar), "h"(mask) : "memory");
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// K-major, 128 B swizzle UMMA operand: 8-row groups 1024 B apart (SBO), version 1 (sm100), layout SWIZZLE_128B.
__device__ __forceinline__ uint64_t sw128_desc(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 46) |
         ((uint64_t)2 << 61);
}
#endif

template <bool kMC>
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
  uint32_t rank = 0;
  if (kMC) asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  if (threadIdx.x == 0) {
    for (int i = 0; i < stages; ++i) {
      mbar_init(smem_u32(&full[i]), 1);
      mbar_init(smem_u32(&empty[i]), kMC ? 2 : 1);
    }
    mbar_init(smem_u32(&accum), 1);
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
  if (kMC) cluster_sync();   // the peer's barriers are initialised before any multicast lands in its smem
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t tmem = tmem_slot;

  if (warp == 0 && lane == 0) {   // TMA producer
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    const uint32_t half = (uint32_t)(Bpad >> 1);
    for (int i = 0; i < nIt; ++i) {
      const int s = i % stages;
      if (i >= stages) mbar_wait(smem_u32(&empty[s]), ((i / stages) - 1) & 1);
      const uint32_t fb = smem_u32(&full[s]);
      const uint32_t a = sbase + s * stageBytes, b = a + kbs * kWBlk;
      // own W bytes plus the full emb stage (both halves, one from each producer when kMC)
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(fb), "r"(stageBytes) : "memory");
      asm volatile(
          "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5}], [%2], %6;"
          ::"r"(a), "l"((uint64_t)&mapW), "r"(fb), "r"(0), "r"(m0), "r"(i * kbs), "l"(pol) : "memory");
      for (int j = 0; j < kbs; ++j) {
        if (kMC) {
          asm volatile(
              "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
              ::"r"(b + j * eBlk + rank * half * 128), "l"((uint64_t)&mapE), "r"(fb), "r"((i * kbs + j) * kBK),
                "r"((int)(rank * half)), "h"((uint16_t)3) : "memory");
        } else {
          asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
                       ::"r"(b + j * eBlk), "l"((uint64_t)&mapE), "r"(fb), "r"((i * kbs + j) * kBK), "r"(0) : "memory");
        }
      }
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
      if (kMC) umma_commit_mc(smem_u32(&empty[s]), 3);
      else umma_commit(smem_u32(&empty[s]));
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
  if (kMC) cluster_sync();   // no CTA exits while its peer's commits can still arrive on its barriers
#endif
}
}  // namespace

extern "C" int adamod_tc_small(const float* emb, const float* W, const float* bias, float* const* outs, int B,
                               cudaStream_t stream) {
  if (B < 1 || B > 256) return 400;
  const bool mc = B > 131;   // emb multicast across 2-CTA clusters (B = 132..256)
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
  // emb rows B..Bpad-1 are out of bounds and zero-filled by TMA; with mc each box is one half of the rows.
  cuuint64_t dE[2] = {(cuuint64_t)kD, (cuuint64_t)B}, sE[1] = {(cuuint64_t)kD * 4};
  cuuint32_t bE[2] = {kBK, (cuuint32_t)(mc ? Bpad / 2 : Bpad)};
  if (cuTensorMapEncodeTiled(&mE, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 2, (void*)emb, dE, sE, bE, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return 402;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(tc_small_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemDyn);
    cudaFuncSetAttribute(tc_small_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemDyn);
    attr_set = true;
  }
  OutPtrs op;
  for (int i = 0; i < 6; ++i) op.p[i] = outs[i];
  const int smem = stages * kbs * perK + 1024;
  if (!mc) {
    tc_small_kernel<false><<<kN / kRows, 128, smem, stream>>>(mW, mE, bias, op, B, Bpad, kbs, stages, tmem_cols);
    return (int)cudaGetLastError();
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(kN / kRows);
  cfg.blockDim = dim3(128);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 2;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, tc_small_kernel<true>, mW, mE, bias, op, B, Bpad, kbs, stages, tmem_cols);
  return (int)cudaGetLastError();
}
```

```cuda file=kernel.cu
#include <cuda_runtime.h>

// One launch per call; the kernel is chosen from B only.
//   B <= 256: hand-written tcgen05 TF32 kernel with evict_first W stream (tc_small.cu); 132..256 multicast emb
//             across 2-CTA clusters.
//   B >  256: CUTLASS sm100 2SM TF32 GEMM with fused bias + six-way epilogue (r1 tiles).
extern "C" int adamod_tc_small(const float*, const float*, const float*, float* const*, int, cudaStream_t);
extern "C" int adamod_gemm_256x192(const float*, const float*, const float*, float* const*, int, int, int, cudaStream_t);
extern "C" int adamod_gemm_256x256(const float*, const float*, const float*, float* const*, int, int, int, cudaStream_t);

extern "C" int adamod_launch(const float* emb, const float* weight, const float* bias, float* const* outs, int B,
                             cudaStream_t stream) {
  if (B <= 256) return adamod_tc_small(emb, weight, bias, outs, B, stream);
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
id: r3-e3-tc05-emb-mcast2
parents:
  - r2-e1-tc05-ef-small
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
  H20 (refuted): for 131 < B <= 256, multicasting each emb k-block across a 2-CTA cluster (each CTA loads half the
  rows and sends them to both) would halve emb L2 traffic and cut cycles by >= 10%. Measured: multicast changes
  cycles by -1..+4% at 132-256. The kernel is bound by per-SM operand ingest (W + full emb landing in each SM's
  shared memory, about 64-75 B/clk/SM), which multicast does not reduce. Only a cta_group::2 pair, where each SM
  receives Bpad/2 emb rows, lowers it.
expected_effect:
  S:
    pct: 0
    confidence: high
  M:
    pct: 1
    confidence: medium
  L:
    pct: 0
    confidence: high
resources_sm100a:
  regs_per_thread: 32
  smem_per_cta_bytes: 230400
  threads_per_cta: 128
  ctas_per_sm: 1
  bytes_in_flight_per_sm: 204800
  launches_per_call: 1
knobs:
  ROWS_PER_CTA: 128
  BK: 32
  KBS_RULE: "unchanged from parent (192/211: 1x5, 256: 1x4)"
  MC_RANGE: "131 < B <= 256, cluster 2x1x1, ctaMask 0b11"
  BPAD: "B rounded to 16 (Bpad/2 is a multiple of 8 rows = whole 1024 B SW128 atoms; rounding to 32 is not needed)"
  W_HINT: evict_first
  TMA_DTYPE: TFLOAT32
dispatch:
  - max_tokens: 131
    kernel: tc_small_kernel<false> (unchanged)
    meta:
      launches: 1
  - max_tokens: 256
    kernel: tc_small_kernel<true> (emb multicast, 72 clusters of 2)
    meta:
      launches: 1
      cluster: 2
  - max_tokens: 384
    kernel: Tf32Gemm<256,192,0>
    meta:
      launches: 1
  - max_tokens: 512
    kernel: Tf32Gemm<256,256,0>
    meta:
      launches: 1
  - max_tokens: null
    kernel: Tf32Gemm<256,192,0>
    meta:
      launches: 1
tests:
  - H20
findings:
  - "H20 refuted (probe, interleaved, harness flush, 3 rounds, rented SM clock 1.66-1.78 GHz from clock64/globaltimer): cycles base vs emb-multicast B=132 74.3-76.2k vs 74.9-75.3k; 192 80.1-80.8k vs 82.0-83.2k (+2-4%); 211 80.9-81.4k vs 82.8-83.5k (+2-3%); 256 85.8-87.6k vs 86.6-87.7k. Per-clock efficiency 2*128*Bpad*3072/(cyc*4096): 132 0.37/0.37, 192 0.46/0.45, 211 0.53/0.52, 256 0.57/0.56 [probe_b200]"
  - "harness_time (same probe) us base vs mc: 132 44.1-44.4 vs 44.6-44.9; 192 48.9-49.0 vs 49.6-49.9; 211 49.7-50.0 vs 50.5-50.6; 256 53.1-53.8 vs 53.5-53.6 [probe_b200]"
  - "kbs=2 at B=192 does not fit 3 stages (2 x 2 stages only): base 2x2 98-105k cycles (57-60 us), mc 2x2 87.5-90.6k (53.6-54.0 us). Multicast cuts cycles 10-15% with a 2-deep ring (emb L2 latency exposed), but both lose to the 1x5 ring [probe_b200]"
  - "limiter isolated (probe, W tile index blockIdx%8 so W is L2-hot, and/or MMAs skipped): B=192/211/256 hot-W 80.7-82.3k cycles vs 85.5-87.5k cold: DRAM is not the M-band limit. Feed only (no MMA, hot W) 62-64k cycles = 3.9-4.6 MB per CTA at 63-74 B/clk/SM; B=16 hot feed-only 23.3k (75 B/clk). MMA ideal 37-49k. So TMA ingest per SM (L2 -> SM bytes per SM clock) bounds the kernel, and MMA + feed overlap only partly (81k vs max(63k, 43k)) [probe_b200]"
  - "consequence: the 1.26 portal/rented ratio at 192/211 (H15) is the SM-clock-bound ingest. A cta_group::2 pair (H19/E1) reduces landing bytes per SM from 16 KB W + Bpad*128 B emb to 16 KB + Bpad*64 B per k-block (-30% at 192, -33% at 256); multicast of emb on top of the pair is not needed [INFERRED from the probes]"
  - "B=132 cold 75k cycles = hot-W 53.5k + DRAM: at 132 the kernel is still DRAM-bound at the rented clock (span 42 us); feed-only cold 79-81k cycles at 1.96 GHz [probe_b200]"
  - "max |err| vs fp64 (W/sqrt(3072), bias randn): B=132 1.45e-3, 192 1.64e-3, 211 1.54e-3, 256 1.49e-3, identical for base and multicast (same operands, same accumulation order) [probe_b200]"
  - "cudaOccupancyMaxActiveClusters = 74 clusters of 2 at 225 KB smem for Bpad 144-256: the 72 clusters fit in one wave [probe_b200]"
  - "run_tests 16/16 PASS. Rented us vs parent same GPU: 131 44.1 vs 44.1, 192 49.4 vs 48.9 (+0.9%), 211 50.7 vs 50.2 (+0.9%); S unchanged; L -1.4% is drift (identical CUTLASS code). Emulator 0.6117 +- 0.012 (P beat 53%), not a real gain [run_tests]"
  - "predicted portal time = max(1.05 x 42 us DRAM part, cycles/1.55 GHz): base 192 86k -> 55.5 us, mc 87.7k -> 56.6 us; base 211 86k -> 55.5, mc 88.5k -> 57.1. The portal measured base 62.2/64.0, so this model underpredicts the parent by about 11% at 192/211 [analysis]"
  - "Bpad rounding: multicast halves only need Bpad/2 to be a multiple of 8 rows (Bpad multiple of 16); 192/211/256 keep Bpad 192/224/256, so the N of the comparison is identical [design]"
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
    rows: 128
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
    note: "tcgen05/TMEM are sm_100a only; the kernel body compiles empty for other targets"
  A100:
    runs: false
    representative: false
  L40S:
    runs: false
    representative: false
risks:
  - "No expected gain: 192/211 are 1-3% slower in cycles than the parent; do not spend a portal slot on it"
  - "Cluster launch for 132-256 adds the cross-CTA empty-barrier coupling (a stall in one CTA stalls its peer)"
  - "Clustered launches slowed later plain launches on #38 (constraint list); another reason not to ship"
measure_first:
  - B200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```