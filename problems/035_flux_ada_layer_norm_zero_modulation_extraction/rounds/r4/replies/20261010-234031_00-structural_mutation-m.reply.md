### Rationale
I built and tested tc_pair2, the H19 kernel for 256 < B ≤ 512 (E1). It passes 16/16 workloads, uses one launch, and its max |err| is 1.51–1.63e-3. Against the r2 reference in run_tests it is faster at 449 (−14.1%) and 512 (−9.2%), but only −2.3/−2.4% at 373/384. Unchanged CUTLASS paths in the same run moved about −2.5%, so the net gain is roughly −11%/−6% at 449/512 and about 0% at 373/384.

Every 1-k-block stage costs a fixed ~575 ns (950–1200 clk, depending on the probe container), even with resident operands. Nothing I tried removes it except kbs=2, which doesn't fit enough stages at Bpad 384. So 449/512 run at 93–100% of the MMA ideal, but 384 is held at that floor. H19 is partly supported. The ≥20%-fewer-cycles criterion is not shown: the CUTLASS clock was never measured, and wall times are within 10% at 373/384/512.

```json solution-spec
{"name": "r4-e1-tc-pair2-twoacc", "definition": "035_flux_ada_layer_norm_zero_modulation_extraction", "author": "solx-loop", "description": "Two-accumulator 2SM tcgen05 TF32 kernel for 256 < B <= 512 (one pass over N = Bpad, kbs1 ring with paired waits); tc_small/tc_pair unchanged; CUTLASS 256x192 for B > 512", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": [], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
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

// 256 < B <= 512: 2SM tcgen05 TF32 GEMM over the whole batch in one pass (N = Bpad = B rounded up to 16).
// A cluster of two CTAs computes a 256-row W tile; the batch is split into two UMMAs per K = 8 step, N1 = 256 and
// N2 = Bpad - 256, accumulating into TMEM columns [0, 256) and [256, Bpad) of a 512-column allocation, so TMEM
// column = batch row. Per k-block each CTA holds its 128 W rows (16 KB) and Bpad / 2 emb rows: 128 rows for UMMA1
// (batch rank * 128 ...) followed by N2 / 2 rows for UMMA2 (batch 256 + rank * N2 / 2 ...), loaded by two emb boxes;
// the second B descriptor is the first plus 16 KB. Only the leader issues MMAs; commits are multicast to both CTAs.
// The ring holds 1-k-block stages (4-5 fit in 226 KB). Every 1-k-block stage pays a fixed floor of about 575 ns
// (950-1200 clk) whatever its MMA work (E1 probe, also with resident operands), so the MMA thread waits for two stages
// at a time and issues 16 UMMAs before the next wait (about 2.5% faster than per-stage waits); it still commits each
// stage separately so that refills stay 1-k-block grained.
// W loads carry the L2 evict_first policy; tensor maps use TFLOAT32 (round to nearest on load).
// Epilogue: 8 warps, warp w reads TMEM lane quadrant w % 4 and every other 16-column chunk, adds bias, six-way split.
namespace {
constexpr int kD = 3072, kN = 6 * kD, kRows = 128, kBK = 32, kKB = kD / kBK, kWBlk = kRows * kBK * 4;
constexpr int kN1 = 256, kE1Blk = (kN1 / 2) * kBK * 4;   // UMMA1 emb rows per CTA: 128 x 128 B = 16 KB
constexpr int kMaxStages = 8, kSmemDyn = 226 * 1024, kThreads = 256, kTmemCols = 512;
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
__device__ __forceinline__ void mma2(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, p; }"
               ::"r"(d), "l"(a), "l"(b), "r"(idesc), "r"(acc));
}
#endif

__global__ void __launch_bounds__(kThreads, 1) tc_pair2_kernel(const __grid_constant__ CUtensorMap mapW,
                                                              const __grid_constant__ CUtensorMap mapE1,
                                                              const __grid_constant__ CUtensorMap mapE2,
                                                              const float* __restrict__ bias, OutPtrs outs, int B,
                                                              int Bpad, int stages) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  extern __shared__ uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t full[kMaxStages], empty[kMaxStages], accum;
  __shared__ uint32_t tmem_slot;
  const uint32_t sbase = (smem_u32(smem_raw) + 1023) & ~1023u;
  const int n2 = Bpad - kN1, n2h = n2 / 2;
  const uint32_t eBlk = kE1Blk + n2h * 128, stageBytes = kWBlk + eBlk;   // per CTA
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const int m0 = blockIdx.x * kRows;
  if (threadIdx.x == 0) {
    for (int i = 0; i < stages; ++i) { mbar_init(smem_u32(&full[i])); mbar_init(smem_u32(&empty[i])); }
    mbar_init(smem_u32(&accum));
    asm volatile("fence.mbarrier_init.release.cluster;");
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapE1) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapE2) : "memory");
  }
  if (warp == 3) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(&tmem_slot)), "r"(kTmemCols));
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
      const uint32_t fl = smem_u32(&full[s]);
      const uint32_t fb = mapa0(fl);   // the leader's full barrier collects the bytes of both CTAs
      if (rank == 0)
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(fl), "r"(2 * stageBytes) : "memory");
      const uint32_t a = sbase + s * stageBytes, b = a + kWBlk;
      asm volatile(
          "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5}], [%2], %6;"
          ::"r"(a), "l"((uint64_t)&mapW), "r"(fb), "r"(0), "r"(m0), "r"(i), "l"(pol) : "memory");
      // emb rows >= B are out of bounds and zero-filled by TMA (their bytes still count toward complete_tx).
      asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
                   ::"r"(b), "l"((uint64_t)&mapE1), "r"(fb), "r"(i * kBK), "r"((int)rank * (kN1 / 2)) : "memory");
      asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
                   ::"r"(b + kE1Blk), "l"((uint64_t)&mapE2), "r"(fb), "r"(i * kBK), "r"(kN1 + (int)rank * n2h) : "memory");
    }
  } else if (warp == 1 && lane == 0 && rank == 0) {   // MMA issuer (leader): 2 x UMMA 256 x N x 8 per K step
    const uint32_t ib = (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)(2 * kRows >> 4) << 24);
    const uint32_t idesc1 = ib | ((uint32_t)(kN1 >> 3) << 17), idesc2 = ib | ((uint32_t)(n2 >> 3) << 17);
    for (int i = 0; i < kKB; ++i) {   // kKB is even: stages are waited for in pairs
      const int s = i % stages;
      if ((i & 1) == 0) {
        mbar_wait(smem_u32(&full[s]), (i / stages) & 1);
        mbar_wait(smem_u32(&full[(i + 1) % stages]), ((i + 1) / stages) & 1);
        asm volatile("tcgen05.fence::after_thread_sync;");
      }
      const uint32_t a = sbase + s * stageBytes, b = a + kWBlk;
      const uint64_t da = sw128_desc(a), db1 = sw128_desc(b), db2 = sw128_desc(b + kE1Blk);
#pragma unroll
      for (int k = 0; k < kBK / 8; ++k) {
        const uint32_t acc = (i | k) != 0;
        mma2(tmem, da + 2 * k, db1 + 2 * k, idesc1, acc);
        mma2(tmem + kN1, da + 2 * k, db2 + 2 * k, idesc2, acc);
      }
      umma_commit2(smem_u32(&empty[s]));
    }
    umma_commit2(smem_u32(&accum));
  }
  __syncwarp();

  mbar_wait(smem_u32(&accum), 0);
  asm volatile("tcgen05.fence::after_thread_sync;");
  const int q = warp & 3, h = warp >> 2;
  const int m = m0 + q * 32 + lane;
  const int c = m0 / kD;
  const float bv = bias[m];
  float* o = outs.p[c] + (m - c * kD);
  for (int c0 = h * 16; c0 < Bpad; c0 += 32) {
    uint32_t r[16];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                 : "r"(tmem + ((uint32_t)(q * 32) << 16) + c0));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
    for (int i = 0; i < 16; ++i)
      if (c0 + i < B) o[(size_t)(c0 + i) * kD] = __uint_as_float(r[i]) + bv;
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  cluster_sync();
  if (warp == 3) {
    asm volatile("tcgen05.fence::after_thread_sync;");
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(kTmemCols));
  }
#endif
}
}  // namespace

extern "C" int adamod_tc_pair2(const float* emb, const float* W, const float* bias, float* const* outs, int B,
                               cudaStream_t stream) {
  if (B <= 256 || B > 512) return 430;
  const int Bpad = (B + 15) / 16 * 16, n2h = (Bpad - kN1) / 2;
  const int perK = kWBlk + kE1Blk + n2h * 128, avail = kSmemDyn - 1024;
  int stages = avail / perK;   // Bpad 272-384: 5, 400-464: 5 (46 KB), 480-512: 4
  if (stages > kMaxStages) stages = kMaxStages;
  if (stages < 2) return 431;
  CUtensorMap mW, mE1, mE2;
  // W viewed as [96 k-blocks][18432 rows][32]: one box = one k-block of 128 rows, a canonical SW128 atom stack.
  cuuint64_t dW[3] = {(cuuint64_t)kBK, (cuuint64_t)kN, (cuuint64_t)kKB}, sW[2] = {(cuuint64_t)kD * 4, (cuuint64_t)kBK * 4};
  cuuint32_t bW[3] = {kBK, kRows, 1}, es[3] = {1, 1, 1};
  if (cuTensorMapEncodeTiled(&mW, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 3, (void*)W, dW, sW, bW, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return 432;
  cuuint64_t dE[2] = {(cuuint64_t)kD, (cuuint64_t)B}, sE[1] = {(cuuint64_t)kD * 4};
  cuuint32_t bE1[2] = {kBK, (cuuint32_t)(kN1 / 2)}, bE2[2] = {kBK, (cuuint32_t)n2h};
  if (cuTensorMapEncodeTiled(&mE1, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 2, (void*)emb, dE, sE, bE1, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return 433;
  if (cuTensorMapEncodeTiled(&mE2, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 2, (void*)emb, dE, sE, bE2, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return 434;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(tc_pair2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemDyn);
    attr_set = true;
  }
  OutPtrs op;
  for (int i = 0; i < 6; ++i) op.p[i] = outs[i];
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(kN / kRows);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = stages * perK + 1024;
  cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 2;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, tc_pair2_kernel, mW, mE1, mE2, bias, op, B, Bpad, stages);
  return (int)cudaGetLastError();
}
```

```cuda file=kernel.cu
#include <cuda_runtime.h>

// One launch per call; the kernel is chosen from B only.
//   B <= 131: hand-written tcgen05 TF32 kernel, one-SM CTAs, evict_first W stream (tc_small.cu).
//   131 < B <= 256: hand-written 2SM tcgen05 TF32 kernel, cluster of 2, 2-k-block stages (tc_pair.cu).
//   256 < B <= 512: hand-written 2SM kernel, whole batch in one pass as two TMEM accumulators (tc_pair2.cu).
//   B >  512: CUTLASS sm100 2SM TF32 GEMM 256x192 with fused bias + six-way epilogue (r1 tile).
extern "C" int adamod_tc_small(const float*, const float*, const float*, float* const*, int, cudaStream_t);
extern "C" int adamod_tc_pair(const float*, const float*, const float*, float* const*, int, cudaStream_t);
extern "C" int adamod_tc_pair2(const float*, const float*, const float*, float* const*, int, cudaStream_t);
extern "C" int adamod_gemm_256x192(const float*, const float*, const float*, float* const*, int, int, int, cudaStream_t);

extern "C" int adamod_launch(const float* emb, const float* weight, const float* bias, float* const* outs, int B,
                             cudaStream_t stream) {
  if (B <= 131) return adamod_tc_small(emb, weight, bias, outs, B, stream);
  if (B <= 256) return adamod_tc_pair(emb, weight, bias, outs, B, stream);
  if (B <= 512) return adamod_tc_pair2(emb, weight, bias, outs, B, stream);
  int device = 0;
  cudaGetDevice(&device);
  static int sms = 0;   // device property, queried once
  if (sms == 0) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
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
id: r4-e1-tc-pair2-twoacc
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
  E1/H19: one cta_group::2 pair per 256 W rows computes the whole batch (N = Bpad <= 512) in one pass as two UMMAs
  (N1 = 256, N2 = Bpad - 256) into two TMEM accumulators, fed by a kbs=1 ring with an evict_first W stream. This removes
  the CUTLASS batch tiling and padding at 373-512. Measured: 449/512 run at 93-100% of the MMA ideal per stage, but at
  Bpad 384 every 1-k-block stage is held at a fixed ~575 ns floor (950-1200 clk, depending on container), 26-30% above
  the 768-clk ideal. Rented: -14.1% at 449, -9.2% at 512, -2.3% at 373/384 vs the CUTLASS tiles (about -11/-6/0% net of
  the -2.5% drift seen on unchanged paths in the same run). Portal by the operator rule (cycles / 1.55 GHz + 11%):
  about 78-79 us at 373/384 (vs 83.0), 80 at 449 (vs 101.4), 83 at 512 (vs 101.9).
expected_effect:
  S:
    pct: 0
    confidence: high
  M:
    pct: -2
    confidence: low
  L:
    pct: -6
    confidence: medium
resources_sm100a:
  regs_per_thread: 40
  smem_per_cta_bytes: 231424
  threads_per_cta: 256
  ctas_per_sm: 1
  bytes_in_flight_per_sm: 204800
  launches_per_call: 1
knobs:
  KBS: 1
  STAGES_RULE: "(226 KB - 1 KB) / (16 KB + Bpad/2 * 128 B): Bpad 384 -> 5, 464 -> 5, 512 -> 4"
  WAIT_GROUP: 2
  N_SPLIT: "N1 = 256, N2 = Bpad - 256 (TMEM cols [0,256) and [256,Bpad), 512 allocated)"
  EPILOGUE_WARPS: 8
  W_HINT: evict_first
  TMA_DTYPE: TFLOAT32
  CLUSTER: 2
  PAIR2_B_RANGE: "256 < B <= 512"
dispatch:
  - max_tokens: 131
    kernel: tc_small_kernel (unchanged r2 path)
    meta:
      launches: 1
  - max_tokens: 256
    kernel: tc_pair_kernel (unchanged r3 path)
    meta:
      launches: 1
  - max_tokens: 512
    kernel: tc_pair2_kernel
    meta:
      launches: 1
  - max_tokens: null
    kernel: Tf32Gemm<256,192,0>
    meta:
      launches: 1
tests:
  - H19
  - H23
  - H24
findings:
  - "run_tests 16/16 PASS (10 rounds each), one launch. Rented us vs r2-e1-tc05-ef-small on the same GPU: 373 67.3 vs 69.0 (-2.4%), 384 67.3 vs 68.9 (-2.3%), 449 70.0 vs 81.5 (-14.1%), 512 74.2 vs 81.8 (-9.2%); 192/211 45.7/48.2 (r3 pair, unchanged); unchanged CUTLASS L paths 691-919 moved -2.2..-3.0% (drift), so the net tc_pair2 gain is about 0/-11/-6% at 384/449/512. Band change S -1.1% M -3.5% L -5.9%; emulator 0.6270 +- 0.012 (no paths were passed to it) [run_tests]"
  - "precision, max |err| vs fp64 (W/sqrt(3072), bias randn, one draw per B): B=257 1.54e-3, 373 1.56e-3, 384 1.54e-3, 449 1.51-1.56e-3, 512 1.63e-3, 0 elements outside atol+rtol; all <= 1.75e-3. The n1 = 192 split gives 1.58e-3 at 373 [probe_b200]"
  - "first build correct: two accumulators at TMEM cols 0 and 256, 512-col cta_group::2 alloc, emb per CTA as 128 rows (rank*128) then N2/2 rows (256 + rank*N2/2) via two TMA maps, second B desc = first + 16 KB, idescs differ only in N, (i|j|k) zero-init for both [probe_b200]"
  - "cycles per k-block (kbs=1, real W from DRAM after a 252 MB zero flush, evict_first, clock64 + globaltimer, rented SM 1.64-1.72 GHz), container 1: Bpad 384 st3/4/5 = 1004/954/956 (ideal 768: 76-80%); 464 st3/4/5 = 1072/954/955 (ideal 928: 97%); 512 st3/4 = 1107/1028 (ideal 1024: 100%). kbs=2 st2: 1127/1216/1265 (exposed latency with 2 stages). Span per CTA: 106.7k (384), 108.2k (449), 116.4k (512) cycles [probe_b200]"
  - "classification (H23/H24): resident operands (no TMA after warm-up), kbs1: 950 (384), 952-955 (449), 1027 (512); resident kbs2 st2: 771 (384) = ideal, 931 (449), 1027 (512). Feed only (no MMA, clock rises to 1.96 GHz): kbs1 st5 767 (384), 795 (449), 870 (512, st4) = 53-58 B/clk/SM, i.e. about 6 TB/s DRAM, not an L2 ingest ceiling. So 384 is chain/floor-bound, 449 is floor and MMA almost equal, 512 is MMA-bound; ingest (H24) is not binding anywhere [probe_b200]"
  - "H23 floor at N >= 384: the kbs=1 per-stage floor is 950-1200 clk depending on the container (990/970/1200 in three later probes), constant at about 575 ns. It is larger than the 551-674 clk seen for 4-UMMA resident groups in r3. It is the same with 4 UMMAs per stage (UMMA2 skipped: 994 vs 995), with N1/N2 = 192/192 vs 256/128 (999 vs 995), and with interleaved or sequential UMMA order (1005). So it does not depend on MMA work or UMMA count [probe_b200]"
  - "dead ends on the floor (Bpad 384, kbs1 st4, clk/k-block in one container): test_wait spin instead of try_wait 971 vs 970; commit every 2nd stage 1060 (worse, coarser refill; resident 970: no gain); idle warps nanosleep on accum 978; epilogue 2 tcgen05.ld per wait 971; removing tcgen05.fence::after_thread_sync 1198 vs 1197; waiting for 3 stages at a time 1161-1241. Only waiting for 2 stages per 16 UMMAs (shipped) helps: 1167 vs 1197 (-2.5%) in the 1200-clk container [probe_b200]"
  - "CUTLASS comparison, interleaved harness_time, same probe (wall us; CUTLASS clock not measured: pynvml missing, nvidia-smi run aborted): 373 mine 68.3 vs 256x192 68.2 / 256x256 79.2; 384 68.4 vs 68.1 / 79.3; 449 70.9 vs 87.2 / 79.7; 512 74.2 vs 87.9 / 80.0. The best CUTLASS tile per B is the r1 table. My kernel ran at 1.64-1.72 GHz (globaltimer), so CUTLASS's 68.1 us at 384 is about 116k cycles at the same clock: about -5% cycles at 384, -11% at 449, -7% at 512 under the same-clock assumption. Success criterion (>= 20% fewer cycles) not shown [probe_b200]"
  - "fixed costs per CTA outside the K loop: first full barrier at 5.8-6.8k clk (cold DRAM W plus setup), epilogue 6.9k (384) / 8.3k (449) / 9.1k (512) clk, about 12-14% of the span. The 8-warp epilogue stores 28-38 MB in about 4-5 us (about 7 TB/s); a TMA-store epilogue is the next lever there [probe_b200]"
  - "targets: <= 100k cycles at 384 missed (106.7-111k); <= 125k at 512 met (116k); per-stage within 15% of MMA ideal met at 449/512, missed at 384 (+26-30%). Refutation clause half-true: the 384 ring is pinned at the floor at every stage count (3-5), 449/512 are not [probe_b200]"
  - "portal prediction (operator rule max(1.05 x DRAM, cycles/1.55 GHz) + 11%): 373/384 108-111k -> 78-79 us (vs 83.0 portal CUTLASS, -5%), 449 112k -> 80 us (vs 101.4, -21%), 512 116k -> 83 us (vs 101.9, -18%). If the floor is in ns (about 575 ns per stage, as the cross-container numbers suggest) and not in clocks, 373/384 do better at the 1.5 GHz lock (about 64-70 us). The tensor-heavy kernel throttles to 1.64-1.75 GHz on the rented card, so its portal/rented ratio should be about 1.1-1.15, not 1.31 [analysis]"
  - "probe infra: probe containers change between calls (/tmp kept in about half the calls); each measurement probe must rebuild CUTLASS (18-39 s for 2 TUs on 8 cores) and the kernel (1.5 s). Once (after 2 CUTLASS rebuilds) harness_time raised 'No timing results for methodology: cupti' [probe_b200]"
  - "open: at Bpad 384 the only way past the floor is more MMA work per wait group with >= 3 stages of refill: a 48-wide k stage (SW128 32 + SW64 16 boxes, 60 KB per CTA, 3 stages, 1152 clk MMA per stage) or a 3-CTA-per-cluster split. The floor source (not commit count, wait count, fence, spin vs suspend, idle warps or UMMA count) is still unknown; it is about 575 ns per 1-k-block stage in this kernel [analysis]"
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
  - "373/384 gain over CUTLASS 256x192 is about 0% rented; the portal gain depends on whether the ~575 ns per-stage floor is time- or clock-bound. If clock-bound and the kernel is fully clock-scaled, 373/384 could land at or slightly above 83 us"
  - "The per-stage floor varied 950-1200 clk between probe containers; the portal card may sit anywhere in that range (449 is floor/MMA-balanced and would lose up to 20% on a 1200-clk card)"
  - "Uses all 227 KB of shared memory at Bpad 400-464 (5 stages of 46 KB + 1 KB alignment slack + static barriers); a toolchain change in static smem could make the launch fail (status 431/launch error, not a wrong result)"
  - "512-column TMEM allocation per CTA requires 1 CTA per SM (guaranteed by shared memory); co-residency with another TMEM user would block in tcgen05.alloc"
  - "No compile_b200 static check was run on the final file set (registers/spills estimated); run_tests built and passed it"
measure_first:
  - B200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```