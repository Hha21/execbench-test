# quack_rmsnorm_fwd_excerpt.py
#
# Source:  https://github.com/Dao-AILab/quack/blob/35266c3298f0e9bf6d5f46c30aace2eaeae517e3/quack/rmsnorm_config.py
#          https://github.com/Dao-AILab/quack/blob/35266c3298f0e9bf6d5f46c30aace2eaeae517e3/quack/rmsnorm.py
#          Write-up: https://sysml.cs.princeton.edu/blogs/memory-bound-kernels.html (Guo, Zadouri, Dao; H100 only)
# Commit:  35266c3298f0 (2026-09-12)
# Licence: Apache-2.0. Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao. Excerpt, unmodified except that
#          lines are elided ("...") and comments starting "# [solx]" are ours.
#
# Why it matters for #38:
# - quack is the reference "speed-of-light" CuTe DSL norm library and runs on SM100. Its forward config uses
#   the SAME ladder on Hopper and Blackwell (no B200-specific forward tuning; only the backward has one).
# - For N = 128 fp32 it picks 128 threads/CTA, 16 threads/row, 128-bit vectors (vecsize = gcd(N, 128/32) = 4),
#   so 8 rows per CTA and 2 x 16 B per thread: half our per-thread bytes. No cache hints anywhere in the norm path.
# - Pattern worth copying: x via cp.async into smem, weights via a synchronous copy issued BETWEEN commit and
#   wait (overlaps the weight fetch with the x fetch), then one reduction and a direct store from registers.
# - It also supports a (b, H, N) per-head layout with a [H, N] weight (grid.z = head), i.e. #38's QK-norm shape.

# ---- quack/rmsnorm_config.py ----
def _for_hopper_fwd(
    N: int, dtype_width: int, arch_major: int, is_layernorm: bool
) -> RmsNormFwdConfig:
    num_threads = 128 if N <= 16 * 1024 else 256

    threads_per_row = 256
    for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
        if N <= limit:
            threads_per_row = threads
            break

    ...

# ---- quack/rmsnorm.py, RMSNorm.__call__ ----
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        self._cap_cluster_n(vecsize)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        ...
        self.kernel(
            mX, mW, mB, mRes, mO, mResO, mRstd, mMean, eps, tiler_mn, tiled_copy, threads_per_row
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, num_heads],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

# ---- quack/rmsnorm.py, RMSNorm.kernel (residual, layernorm and reload branches elided) ----
        row = tXcX[0][0]
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
            if const_expr(mRes is not None):
                copy(tXgRes, tXsRes, is_async=True)
        cute.arch.cp_async_commit_group()

        ...
        if const_expr(not self.delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)
            if const_expr(mB is not None):
                copy(tXgB, tXrB)

        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)
        ...
            # RMSNorm: compute sum of squares directly
            mean = const_expr(0.0)
            sum_sq_x = row_reduce(
                x * x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                init_val=0.0,
                hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
            )
            rstd = cute.math.rsqrt(sum_sq_x / shape[1] + eps, fastmath=True)
        ...
        x_hat = (x - mean) * rstd if const_expr(self.is_layernorm) else x * rstd
        y = x_hat
        if const_expr(mW is not None):
            w = tXrW.load().to(cute.Float32)
            ...
            y *= w
        ...
        tXrO.store(y.to(tXrO.element_type))
        if row < shape[0]:
            copy(tXrO, tXgO)
