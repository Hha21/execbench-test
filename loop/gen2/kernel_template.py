"""Generation-2 kernels for SOL-ExecBench #38: per-head RMSNorm of Q and K, fp32, [B, S, 48, 128].

Four kernel families. run() picks one per input size from BANDS, which make_gen2.py rewrites for each
candidate. Every family is correct for every shape (ROWS and HB divide 48, so no masks are needed).

  oneshot    one program per ROWS-row tile, Q and K in one launch; for small inputs (no loop, no prologue)
  wstat      persistent, each program owns one (stream, head group) and loads its weights once
  tma        persistent, tiles moved by TMA through host-side tensor descriptors, software-pipelined
  tma_wstat  tma + weight-stationary

Persistent grids are sized from the compiled kernel's real register and shared-memory use on the current GPU (a
compile-only warm-up before the first launch), so a grid never asks for more resident programs than fit, which was
the v028 failure on B200.
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

D = 128
H = 48

# --- candidate config (rewritten by make_gen2.py) ---
BANDS = [(1 << 62, "oneshot", {"ROWS": 16, "num_warps": 8, "EVICT": True})]
# ------------------------------------------------------


@triton.jit
def _qk_oneshot(q_ptr, k_ptr, wq_ptr, wk_ptr, qo_ptr, ko_ptr, n_rows, eps,
                ROWS: tl.constexpr, D: tl.constexpr, H: tl.constexpr, EVICT: tl.constexpr):
    pid = tl.program_id(0)
    n_tiles = n_rows // ROWS
    is_k = pid >= n_tiles
    tile = tl.where(is_k, pid - n_tiles, pid)
    x_ptr = tl.where(is_k, k_ptr, q_ptr)
    w_ptr = tl.where(is_k, wk_ptr, wq_ptr)
    y_ptr = tl.where(is_k, ko_ptr, qo_ptr)
    rows = tile * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, D)
    offs = rows[:, None] * D + cols[None, :]
    w_offs = (rows % H)[:, None] * D + cols[None, :]
    if EVICT:
        x = tl.load(x_ptr + offs, eviction_policy="evict_first")
        w = tl.load(w_ptr + w_offs, eviction_policy="evict_last")
    else:
        x = tl.load(x_ptr + offs)
        w = tl.load(w_ptr + w_offs)
    inv = tl.math.rsqrt(tl.sum(x * x, axis=1) * (1.0 / D) + eps)
    y = (x * inv[:, None]) * w
    if EVICT:
        tl.store(y_ptr + offs, y, eviction_policy="evict_first")
    else:
        tl.store(y_ptr + offs, y)


@triton.jit
def _qk_wstat(q_ptr, k_ptr, wq_ptr, wk_ptr, qo_ptr, ko_ptr, n_tok, eps, n_prog,
              HB: tl.constexpr, D: tl.constexpr, H: tl.constexpr, EVICT: tl.constexpr, STAGES: tl.constexpr):
    # n_prog is a multiple of 2 * (H // HB); program pid always handles the same (stream, head group).
    pid = tl.program_id(0)
    G = H // HB
    grp = pid % (2 * G)
    is_k = grp >= G
    hg = tl.where(is_k, grp - G, grp)
    x_ptr = tl.where(is_k, k_ptr, q_ptr)
    w_ptr = tl.where(is_k, wk_ptr, wq_ptr)
    y_ptr = tl.where(is_k, ko_ptr, qo_ptr)
    heads = hg * HB + tl.arange(0, HB)
    cols = tl.arange(0, D)
    w = tl.load(w_ptr + heads[:, None] * D + cols[None, :])          # once per program
    for tok in tl.range(pid // (2 * G), n_tok, n_prog // (2 * G), num_stages=STAGES):
        offs = (tok * H + heads)[:, None] * D + cols[None, :]
        if EVICT:
            x = tl.load(x_ptr + offs, eviction_policy="evict_first")
        else:
            x = tl.load(x_ptr + offs)
        inv = tl.math.rsqrt(tl.sum(x * x, axis=1) * (1.0 / D) + eps)
        y = (x * inv[:, None]) * w
        if EVICT:
            tl.store(y_ptr + offs, y, eviction_policy="evict_first")
        else:
            tl.store(y_ptr + offs, y)


@triton.jit
def _qk_tma(qd, kd, qod, kod, wq_ptr, wk_ptr, n_rows, eps, n_prog,
            ROWS: tl.constexpr, D: tl.constexpr, H: tl.constexpr, STAGES: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, D)
    for t in tl.range(pid, n_rows // ROWS, n_prog, num_stages=STAGES):
        r0 = t * ROWS
        w_offs = ((r0 + tl.arange(0, ROWS)) % H)[:, None] * D + cols[None, :]
        x = qd.load([r0, 0])
        inv = tl.math.rsqrt(tl.sum(x * x, axis=1) * (1.0 / D) + eps)
        qod.store([r0, 0], (x * inv[:, None]) * tl.load(wq_ptr + w_offs))
        x = kd.load([r0, 0])
        inv = tl.math.rsqrt(tl.sum(x * x, axis=1) * (1.0 / D) + eps)
        kod.store([r0, 0], (x * inv[:, None]) * tl.load(wk_ptr + w_offs))


@triton.jit
def _tma_rows(xd, yd, w, hg, n_tok, eps, start, step,
              HB: tl.constexpr, D: tl.constexpr, H: tl.constexpr, STAGES: tl.constexpr):
    for tok in tl.range(start, n_tok, step, num_stages=STAGES):
        r0 = tok * H + hg * HB
        x = xd.load([r0, 0])
        inv = tl.math.rsqrt(tl.sum(x * x, axis=1) * (1.0 / D) + eps)
        yd.store([r0, 0], (x * inv[:, None]) * w)


@triton.jit
def _qk_tma_wstat(qd, kd, qod, kod, wq_ptr, wk_ptr, n_tok, eps, n_prog,
                  HB: tl.constexpr, D: tl.constexpr, H: tl.constexpr, STAGES: tl.constexpr):
    pid = tl.program_id(0)
    G = H // HB
    grp = pid % (2 * G)
    is_k = grp >= G
    hg = tl.where(is_k, grp - G, grp)
    cols = tl.arange(0, D)
    heads = hg * HB + tl.arange(0, HB)
    w = tl.load(tl.where(is_k, wk_ptr, wq_ptr) + heads[:, None] * D + cols[None, :])   # once per program
    start = pid // (2 * G)
    step = n_prog // (2 * G)
    if is_k:  # uniform per program; each branch has its own pipelined loop
        _tma_rows(kd, kod, w, hg, n_tok, eps, start, step, HB, D, H, STAGES)
    else:
        _tma_rows(qd, qod, w, hg, n_tok, eps, start, step, HB, D, H, STAGES)


# ----------------------------------------------------------------------------- launch helpers

_DEV = None
_FIT = {}  # (family, meta) -> resident programs per SM; depends only on the compiled kernel and the device


def _device():
    global _DEV
    if _DEV is None:
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        _DEV = dict(sms=p.multi_processor_count,
                    threads=getattr(p, "max_threads_per_multi_processor", 2048),
                    regs=getattr(p, "regs_per_multiprocessor", 65536),
                    smem=getattr(p, "shared_memory_per_multiprocessor", 232448))
    return _DEV


def _resident(ck, num_warps):
    """Programs of this compiled kernel that fit on one SM at once."""
    dev = _device()
    threads = 32 * num_warps
    regs_per_warp = -(-max(ck.n_regs, 1) // 8) * 8 * 32
    by_regs = (dev["regs"] // regs_per_warp) // num_warps
    smem = ck.metadata.shared
    by_smem = dev["smem"] // (smem + 1024) if smem else 32
    return max(1, min(32, dev["threads"] // threads, by_regs, by_smem))


def _persistent(key, kernel, args, kwargs, total_units, num_warps, cap, multiple=1):
    """Launch a persistent kernel with as many programs as fit on the SMs at once (at most `cap` per SM).

    The fit is computed once per (family, meta) from a compile-only warm-up of the same kernel, before the first
    launch, so every launch, including the first, uses the same grid.
    """
    sms = _device()["sms"]
    if key not in _FIT:
        n0 = max(multiple, sms * cap // multiple * multiple)
        ck = kernel.warmup(*args(n0), grid=(n0,), **kwargs)
        ck._init_handles()
        _FIT[key] = _resident(ck, num_warps)
    n_prog = max(multiple, min(total_units, sms * min(cap, _FIT[key])) // multiple * multiple)
    kernel[(n_prog,)](*args(n_prog), **kwargs)


def run(query, key, weight_q, weight_k, eps, query_norm, key_norm):
    n_rows = query.numel() // D
    n_tok = n_rows // H
    eps = float(eps)
    family, meta = next((f, m) for limit, f, m in BANDS if n_tok <= limit)
    nw = meta["num_warps"]
    mkey = (family, tuple(sorted(meta.items())))

    if family == "oneshot":
        rows = meta["ROWS"]
        _qk_oneshot[(2 * (n_rows // rows),)](query, key, weight_q, weight_k, query_norm, key_norm, n_rows, eps,
                                             ROWS=rows, D=D, H=H, EVICT=meta["EVICT"], num_warps=nw)
    elif family == "wstat":
        hb = meta["HB"]
        g2 = 2 * (H // hb)
        _persistent(mkey, _qk_wstat,
                    lambda n: (query, key, weight_q, weight_k, query_norm, key_norm, n_tok, eps, n),
                    dict(HB=hb, D=D, H=H, EVICT=meta["EVICT"], STAGES=meta["STAGES"], num_warps=nw),
                    g2 * n_tok, nw, meta["PROGS_PER_SM"], multiple=g2)
    elif family in ("tma", "tma_wstat"):
        rows = meta["ROWS"] if family == "tma" else meta["HB"]
        descs = [TensorDescriptor.from_tensor(t.view(-1, D), [rows, D]) for t in (query, key, query_norm, key_norm)]
        if family == "tma":
            _persistent(mkey, _qk_tma,
                        lambda n: (*descs, weight_q, weight_k, n_rows, eps, n),
                        dict(ROWS=rows, D=D, H=H, STAGES=meta["STAGES"], num_warps=nw),
                        n_rows // rows, nw, meta["PROGS_PER_SM"])
        else:
            g2 = 2 * (H // rows)
            _persistent(mkey, _qk_tma_wstat,
                        lambda n: (*descs, weight_q, weight_k, n_tok, eps, n),
                        dict(HB=rows, D=D, H=H, STAGES=meta["STAGES"], num_warps=nw),
                        g2 * n_tok, nw, meta["PROGS_PER_SM"], multiple=g2)
    else:
        raise ValueError(family)
