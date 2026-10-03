"""Per-head RMSNorm of Q and K (SOL-ExecBench L1/038_flux_multi_head_rmsnorm_qk).

One Triton kernel with tuning knobs. make_variants.py rewrites the knob block below
to produce each variant, so every variant is a self-contained solution file.
"""

import torch
import triton
import triton.language as tl

# --- variant knobs (rewritten by make_variants.py) ---
ROWS = 8  # rows of head_dim=128 handled per tile
NUM_WARPS = 4
NUM_STAGES = 2  # software pipelining depth; only matters for the persistent loop
FUSE = True  # one launch for Q and K together, else one launch each
PERSIST = False  # fixed grid that loops over tiles, else one program per tile
PROGS_PER_SM = 4  # persistent grid size = SMs * PROGS_PER_SM
EVICT = False  # stream-through cache hint on loads/stores
# ------------------------------------------------------

D = 128
H = 48


@triton.jit
def _qk_norm_kernel(
    q_ptr, k_ptr, wq_ptr, wk_ptr, qo_ptr, ko_ptr,
    n_rows, eps, n_prog,
    ROWS: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
    EVICT: tl.constexpr, FUSE: tl.constexpr, PERSIST: tl.constexpr,
):
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(n_rows, ROWS)
    if FUSE:
        total = n_tiles * 2
    else:
        total = n_tiles
    if PERSIST:
        step = n_prog
    else:
        step = total  # loop body runs exactly once per program
    cols = tl.arange(0, D)
    for t in range(pid, total, step):
        is_k = t >= n_tiles
        tile = tl.where(is_k, t - n_tiles, t)
        x_ptr = tl.where(is_k, k_ptr, q_ptr)
        w_ptr = tl.where(is_k, wk_ptr, wq_ptr)
        y_ptr = tl.where(is_k, ko_ptr, qo_ptr)
        rows = tile * ROWS + tl.arange(0, ROWS)
        mask = (rows < n_rows)[:, None]
        offs = rows[:, None] * D + cols[None, :]
        if EVICT:
            x = tl.load(x_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_first")
        else:
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        w = tl.load(w_ptr + (rows % H)[:, None] * D + cols[None, :], mask=mask, other=0.0)
        inv = tl.math.rsqrt(tl.sum(x * x, axis=1) / D + eps)
        y = (x * inv[:, None]) * w
        if EVICT:
            tl.store(y_ptr + offs, y, mask=mask, eviction_policy="evict_first")
        else:
            tl.store(y_ptr + offs, y, mask=mask)


_NUM_SMS = None


def _num_sms():
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return _NUM_SMS


def run(query, key, weight_q, weight_k, eps, query_norm, key_norm):
    n_rows = query.numel() // D
    n_tiles = triton.cdiv(n_rows, ROWS)
    eps = float(eps)
    if FUSE:
        launches = [(query, key, weight_q, weight_k, query_norm, key_norm, 2 * n_tiles)]
    else:
        launches = [
            (query, query, weight_q, weight_q, query_norm, query_norm, n_tiles),
            (key, key, weight_k, weight_k, key_norm, key_norm, n_tiles),
        ]
    for q, k, wq, wk, qo, ko, total in launches:
        n_prog = min(total, _num_sms() * PROGS_PER_SM) if PERSIST else total
        _qk_norm_kernel[(n_prog,)](
            q, k, wq, wk, qo, ko, n_rows, eps, n_prog,
            ROWS=ROWS, D=D, H=H, EVICT=EVICT, FUSE=FUSE, PERSIST=PERSIST,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )
