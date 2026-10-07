# flashinfer_qk_rmsnorm_sm100_excerpt.py
#
# Source:  https://github.com/flashinfer-ai/flashinfer/blob/b6b4f1ec35c40995b9901fba8fb93267c45ffb6b/flashinfer/norm/kernels/rmsnorm.py
#          (class QKRMSNormKernel; Blackwell rule added by PR #5305, merged 2026-09-18:
#          https://github.com/flashinfer-ai/flashinfer/pull/5305)
# Commit:  b6b4f1ec35c4 (main, 2026-10-07)
# Licence: Apache-2.0. Copyright (c) 2025 by FlashInfer team. Excerpt, unmodified except that
#          lines are elided ("...") and comments starting "# [solx]" are ours.
#
# Why it matters for #38 (per-head RMSNorm, head_dim 128):
# - This is NVIDIA-adjacent production code for exactly our op shape (one row = one (token, head)).
# - On SM100/103/107 it deliberately uses FEWER threads per row (4 instead of 16 for head_dim 128),
#   so each thread owns ~32 elements: "16 B per thread ... is only 32 KB in flight per SM".
#   PR #5305 measured 1.14x (M=8192) and 1.20x (M=32768) on B200 for head_dim 128, neutral for M <= 512.
# - Loads go through cp.async (128-bit, LDGSTS) into smem, then smem -> registers; weights are a
#   plain synchronous copy issued between commit and wait; stores are plain 128-bit. No cache hints.
# - For fp32 head_dim 128 on B200 this gives: 128-thread CTA, 32 rows (16 KB) per CTA, 128 B per thread.


# Architectures where the norm kernels are bound by bytes in flight per SM;
# the tiling adjustments below apply only to these.
_LATENCY_BOUND_SMS = (100, 103, 107)

    ...

class QKRMSNormKernel:
    ...
        elem_bytes = dtype.width // 8
        max_vec_size = COPY_BITS // 8 // elem_bytes

        h_align = head_dim & (-head_dim)
        self.vec_size = min(h_align, max_vec_size)
        self.copy_bits = self.vec_size * dtype.width

        self.threads_per_row = self._compute_threads_per_row(head_dim, self.sm_version)
        self.num_threads = RMSNormKernel._compute_num_threads(head_dim)
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)

        self.num_vec_blocks = max(
            1,
            (head_dim // self.vec_size + self.threads_per_row - 1)
            // self.threads_per_row,
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

        if self.copy_bits >= 32:
            tile_bytes = self.rows_per_block * self.cols_per_tile * elem_bytes
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            self.use_async_copy = tile_bytes <= props.shared_memory_per_block_optin // 2
        else:
            self.use_async_copy = False

    @staticmethod
    def _compute_threads_per_row(head_dim: int, sm_version: int) -> int:
        """Threads cooperating on one (batch, head) row.

        The shared RMSNorm table leaves each thread with a single 16-byte vector
        for small head_dim, which is too little in flight per SM on Blackwell and
        Rubin. There, use fewer threads per row so each thread owns ~32 elements.
        """
        default = RMSNormKernel._compute_threads_per_row(head_dim)
        if sm_version not in _LATENCY_BOUND_SMS:
            return default
        target = max(head_dim // 32, 1)
        target = 1 << (target.bit_length() - 1)  # power of two for warp shuffles
        return min(default, max(4, target))

    ...
    # [solx] fp32, head_dim 128, sm_100: vec_size 4 (COPY_BITS = 128), threads_per_row 4, num_threads 128,
    # [solx] rows_per_block 32, num_vec_blocks 8. make_tv_layout (norm/utils.py) gives thread t of a row the
    # [solx] columns [4t, 4t+4) + 16k, k = 0..7: each 128-bit instruction covers a 64 B piece of 8 different
    # [solx] rows per warp (not one contiguous 512 B row as in our half-warp layout).
    # [solx] inside the @cute.kernel body:
        # ===== Pass 1: Load input + compute sum of squares =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
        else:
            ...  # [solx] synchronous-copy fallback elided

        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = row_reduce_sum_multirow(
            x_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(head_dim)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        cute.arch.barrier()

        # ===== Pass 2: Normalize and store output =====
        # Re-load x from shared memory to relieve register pressure.
        # Without this, x (up to 128 FP32 values/thread at large H) must
        # survive across the reduction + barrier, causing spills to local mem.
        if cutlass.const_expr(self.use_async_copy):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

        w = tXrW.load().to(Float32)
        y = x * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mY.element_type))

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)
