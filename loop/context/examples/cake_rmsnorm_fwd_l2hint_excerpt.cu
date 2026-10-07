// cake_rmsnorm_fwd_l2hint_excerpt.cu
//
// Source:  https://github.com/flashinfer-ai/flashinfer/blob/b6b4f1ec35c40995b9901fba8fb93267c45ffb6b/csrc/cake_rmsnorm_train/cake_rmsnorm_train_39325dca02d9cc2ad87d_kernel.cu
//          (generated "CAKE" program; role fwd_main for H=512 bf16, T > 16384, sm_100a/sm_103a; added by
//          PR #5741, merged 2026-10-05: https://github.com/flashinfer-ai/flashinfer/pull/5741)
//          Policy constants: https://github.com/NVIDIA/cutlass/blob/0b55a2f691d69981583568fd9eb69687b1f0de8a/include/cute/arch/copy_sm90_desc.hpp
// Commit:  FlashInfer b6b4f1ec35c4 (2026-10-07); CUTLASS 0b55a2f691d6 (2026-09-23)
// Licence: FlashInfer file: Apache-2.0, Copyright (c) 2026 by FlashInfer team. CUTLASS lines: BSD-3-Clause,
//          Copyright (c) NVIDIA CORPORATION & AFFILIATES. Excerpt, unmodified except that lines are elided
//          ("...") and comments starting "// [solx]" are ours.
//
// Why it matters for #38:
// - Per-row knobs of these programs ("L2 load hints, register cache form, rows per CTA, ...") were chosen
//   from paired CUPTI cold-L2 measurements on B200/GB300 (PR #5741). The shipped choice for the streamed input
//   is L2 evict_first on LOADS ONLY (policy constant, no createpolicy instruction), weights via ld.global.nc,
//   and PLAIN stores. This matches our portal data: hinted stores cost ~5% at L (see b200_sota.md section 3).
// - Shape is close to ours: one warp per 1 KB row (we have a half-warp per 512 B row), 64-thread CTAs,
//   2 rows per CTA, one-shot grid (T/2 CTAs). B200: T=65536 (134 MB) in 23.36 us = 5.75 TB/s.
// - The same PR reports 256-bit single loads "not faster" than 128-bit for the H=512 programs on SM100a.

// ---- CUTLASS include/cute/arch/copy_sm90_desc.hpp (enum CacheHintSm90) ----
//   EVICT_NORMAL = 0x1000000000000000,
//   EVICT_FIRST  = 0x12F0000000000000,
//   EVICT_LAST   = 0x14F0000000000000,

// ---- FlashInfer CAKE forward program ----
#define THREADS 64
#define ROWS_PER_CTA 2
#define GROUP_THREADS 32
#define LAUNCH_BOUNDS_THREADS 64
...
__global__ __launch_bounds__(LAUNCH_BOUNDS_THREADS) void
kernel_cake_rmsnorm_train_39325dca02d9cc2ad87d(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ u, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ h_new, float* __restrict__ r, int T, long long x_stride, long long u_stride, float eps)
{
    ...
    #pragma unroll 1
    for (int row0 = bid * ROWS_PER_CTA; row0 < T; row0 += num_bids * ROWS_PER_CTA) {
        int row = row0 + group;
        long long row64 = (long long)row;
        long long xbase = row64 * x_stride;
        long long obase = row64 * 512;
        float x_cache[16];
        float w_cache[16];
        float sum_sq = 0.0f;
        if (row < T) {
            // [solx] weights: read-only path (LDG.E.CONSTANT), no L2 hint
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_vld_0[_blk].x), "=r"(_vld_0[_blk].y), "=r"(_vld_0[_blk].z), "=r"(_vld_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)) : "memory");
            ...
            // [solx] streamed input: L2::cache_hint with the EVICT_FIRST policy constant
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vld_1[_blk].x), "=r"(_vld_1[_blk].y), "=r"(_vld_1[_blk].z), "=r"(_vld_1[_blk].w) : "l"((const void*)(_vptr_1 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
            ...
        }
        float _warp_reduce_0 = sum_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        sum_sq = _warp_reduce_0;
        float total = sum_sq;
        float mean = total * 0.001953125f;
        float _sqrt_0;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_0) : "f"(mean + eps));
        float _rcp_0 = __frcp_rn(_sqrt_0);
        float rstd = _rcp_0;
        ...
                    // [solx] output: plain 128-bit store, default L2 policy
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(y + (obase + (long long)(gtid * 8 + v_3 * (GROUP_THREADS * 8)))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
    ...
