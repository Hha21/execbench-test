"""Compile the exact Triton snippets quoted in playbook_membound.md §3a and §3d for sm_100a/sm_90a."""
import json, sys
import triton, triton.language as tl
sys.path.insert(0, ".")
from probe1 import sass_summary, compile_one
D, H = 128, 48

@triton.jit
def _qk_rms(q_ptr, k_ptr, wq_ptr, wk_ptr, qo_ptr, ko_ptr, n_rows, eps,
            ROWS: tl.constexpr, D: tl.constexpr, H: tl.constexpr):
    pid = tl.program_id(0)
    n_tiles = n_rows // ROWS                 # ROWS divides 48 -> no tail
    is_k = pid >= n_tiles
    tile = tl.where(is_k, pid - n_tiles, pid)
    x_ptr = tl.where(is_k, k_ptr, q_ptr)
    w_ptr = tl.where(is_k, wk_ptr, wq_ptr)
    y_ptr = tl.where(is_k, ko_ptr, qo_ptr)
    rows = tile * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, D)
    offs = rows[:, None] * D + cols[None, :]
    x = tl.load(x_ptr + offs, eviction_policy="evict_first")
    w = tl.load(w_ptr + (rows % H)[:, None] * D + cols[None, :], eviction_policy="evict_last")
    inv = tl.math.rsqrt(tl.sum(x * x, axis=1) * (1.0 / D) + eps)
    tl.store(y_ptr + offs, (x * inv[:, None]) * w, eviction_policy="evict_first")

@triton.jit
def _qk_rms_tma(qd, kd, qod, kod, wq_ptr, wk_ptr, n_rows, eps, n_prog,
                ROWS: tl.constexpr, D: tl.constexpr, H: tl.constexpr, STAGES: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, D)
    for t in tl.range(pid, n_rows // ROWS, n_prog, num_stages=STAGES):
        r0 = t * ROWS
        hq = ((r0 + tl.arange(0, ROWS)) % H)[:, None] * D + cols[None, :]
        x = qd.load([r0, 0])
        inv = tl.math.rsqrt(tl.sum(x * x, axis=1) * (1.0 / D) + eps)
        qod.store([r0, 0], (x * inv[:, None]) * tl.load(wq_ptr + hq))
        x = kd.load([r0, 0])
        inv = tl.math.rsqrt(tl.sum(x * x, axis=1) * (1.0 / D) + eps)
        kod.store([r0, 0], (x * inv[:, None]) * tl.load(wk_ptr + hq))

P6 = ["q_ptr", "k_ptr", "wq_ptr", "wk_ptr", "qo_ptr", "ko_ptr"]
probes = []
for rows, w in [(4, 2), (8, 4), (16, 4)]:
    probes.append((f"playbook_3a_R{rows}_W{w}", _qk_rms, {n: "*fp32" for n in P6} | {"n_rows": "i32", "eps": "fp32"},
                   dict(ROWS=rows, D=D, H=H), dict(num_warps=w), P6 + ["n_rows"]))
for rows, st in [(8, 3), (16, 4)]:
    td = f"tensordesc<fp32[{rows},128]>"
    probes.append((f"playbook_3d_host_R{rows}_S{st}", _qk_rms_tma,
                   {"qd": td, "kd": td, "qod": td, "kod": td, "wq_ptr": "*fp32", "wk_ptr": "*fp32", "n_rows": "i32", "eps": "fp32", "n_prog": "i32"},
                   dict(ROWS=rows, D=D, H=H, STAGES=st), dict(num_warps=4), ["wq_ptr", "wk_ptr", "n_rows"]))
out = []
for name, fn, sig, ce, opts, ptrs in probes:
    for arch, cc in {"sm_100a": 100, "sm_90a": 90}.items():
        rec = dict(probe=name, arch=arch)
        try:
            ck = compile_one(fn, sig, ce, cc, opts, ptrs)
            r = sass_summary(ck.asm["cubin"]); rec.update(r); rec["meta_shared"] = ck.metadata.shared
            rec["div_full"] = ck.asm["ptx"].count("div.full.f32")
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out.append(rec)
        print(json.dumps({k: rec.get(k) for k in ("probe", "arch", "regs", "meta_shared", "local", "div_full", "error")}),
              {k: v for k, v in rec.get("ops", {}).items() if k.startswith(("LDG", "STG", "UTMA", "LDGSTS", "MUFU"))})
open("probe7_results.jsonl", "w").write("\n".join(json.dumps(r) for r in out) + "\n")
