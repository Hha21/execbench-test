"""Compile-only probes, batch 2: PDL hooks, warp_specialize on a non-matmul loop, weight-stationary
persistent loop. The Gluon part here fails (gl.load result has no layout); probe4.py is the working Gluon ring.
Run: python probe2.py            (Triton probes only)
     python probe2.py with_gluon (also the failing Gluon attempt)"""
import json, sys, traceback
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.language.extra.cuda import gdc_wait, gdc_launch_dependents

sys.path.insert(0, ".")
from probe1 import sass_summary, ptx_hints, compile_one  # noqa: E402

D, H = 128, 48


@triton.jit
def k_gdc(q_ptr, qo_ptr, n_rows, eps, ROWS: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, D)
    offs = rows[:, None] * D + cols[None, :]
    gdc_wait()
    x = tl.load(q_ptr + offs)
    gdc_launch_dependents()
    inv = tl.math.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    tl.store(qo_ptr + offs, x * inv[:, None])


@triton.jit
def k_tma_ws(qd, qod, wq_ptr, n_rows, eps, n_prog, ROWS: tl.constexpr, D: tl.constexpr, H: tl.constexpr, STAGES: tl.constexpr):
    pid = tl.program_id(0)
    n_tiles = n_rows // ROWS
    cols = tl.arange(0, D)
    for t in tl.range(pid, n_tiles, n_prog, num_stages=STAGES, warp_specialize=True):
        r0 = t * ROWS
        rows = r0 + tl.arange(0, ROWS)
        x = qd.load([r0, 0])
        inv = tl.math.rsqrt(tl.sum(x * x, axis=1) / D + eps)
        qod.store([r0, 0], (x * inv[:, None]) * tl.load(wq_ptr + (rows % H)[:, None] * D + cols[None, :]))


@triton.jit
def k_wstat(q_ptr, k_ptr, wq_ptr, wk_ptr, qo_ptr, ko_ptr, n_tok, eps, n_prog,
            HB: tl.constexpr, D: tl.constexpr, H: tl.constexpr):
    # Weight-stationary persistent loop: tile = HB heads of one token for Q or K. Program p always
    # handles the same (tensor, head-group) so its weight tile is loaded once, outside the loop.
    # n_prog must be a multiple of G2 = 2 * H // HB.
    G: tl.constexpr = H // HB
    G2: tl.constexpr = 2 * G
    pid = tl.program_id(0)
    grp = pid % G2
    is_k = grp >= G
    hg = tl.where(is_k, grp - G, grp)
    x_ptr = tl.where(is_k, k_ptr, q_ptr)
    w_ptr = tl.where(is_k, wk_ptr, wq_ptr)
    y_ptr = tl.where(is_k, ko_ptr, qo_ptr)
    heads = hg * HB + tl.arange(0, HB)
    cols = tl.arange(0, D)
    w = tl.load(w_ptr + heads[:, None] * D + cols[None, :])
    step = n_prog // G2
    for tok in range(pid // G2, n_tok, step):
        offs = (tok * H + heads)[:, None] * D + cols[None, :]
        x = tl.load(x_ptr + offs, eviction_policy="evict_first")
        inv = tl.math.rsqrt(tl.sum(x * x, axis=1) / D + eps)
        tl.store(y_ptr + offs, (x * inv[:, None]) * w, eviction_policy="evict_first")


def run_triton_probes(out):
    probes = [
        ("gdc_pdl", k_gdc, {"q_ptr": "*fp32", "qo_ptr": "*fp32", "n_rows": "i32", "eps": "fp32"}, dict(ROWS=8, D=D),
         dict(num_warps=4, launch_pdl=True), ["q_ptr", "qo_ptr"]),
        ("tma_ws_R8_S3", k_tma_ws, {"qd": "tensordesc<fp32[8,128]>", "qod": "tensordesc<fp32[8,128]>", "wq_ptr": "*fp32",
                                    "n_rows": "i32", "eps": "fp32", "n_prog": "i32"},
         dict(ROWS=8, D=D, H=H, STAGES=3), dict(num_warps=4), ["wq_ptr", "n_rows"]),
        ("wstat_HB16", k_wstat, {"q_ptr": "*fp32", "k_ptr": "*fp32", "wq_ptr": "*fp32", "wk_ptr": "*fp32", "qo_ptr": "*fp32",
                                 "ko_ptr": "*fp32", "n_tok": "i32", "eps": "fp32", "n_prog": "i32"},
         dict(HB=16, D=D, H=H), dict(num_warps=4), ["q_ptr", "k_ptr", "wq_ptr", "wk_ptr", "qo_ptr", "ko_ptr"]),
    ]
    for name, fn, sig, ce, opts, ptrs in probes:
        for arch, cc in {"sm_100a": 100, "sm_90a": 90}.items():
            rec = dict(probe=name, arch=arch)
            try:
                ck = compile_one(fn, sig, ce, cc, opts, ptrs)
                rec.update(sass_summary(ck.asm["cubin"]))
                rec["ptx_hints"] = ptx_hints(ck.asm["ptx"])
                rec["meta_shared"] = ck.metadata.shared
                rec["num_warps_meta"] = ck.metadata.num_warps
                rec["launch_pdl_meta"] = getattr(ck.metadata, "launch_pdl", None)
                if arch == "sm_100a":
                    Path(f"{name}.{arch}.ptx").write_text(ck.asm["ptx"])
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:600]}"
            out.append(rec)
            print(json.dumps(rec), flush=True)


# ---------------------------------------------------------------- Gluon explicit TMA ring
def run_gluon_probe(out):
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared
    from triton.experimental.gluon._runtime import GluonASTSource
    from triton.experimental.gluon.language._layouts import NVMMASharedLayout

    @gluon.jit
    def g_rms(x_desc, y_desc, w_ptr, n_rows, eps, n_prog,
              ROWS: gl.constexpr, D: gl.constexpr, H: gl.constexpr, STAGES: gl.constexpr):
        # One tensor (Q or K) per launch for simplicity; persistent; STAGES-deep TMA load ring,
        # double-buffered TMA store.
        NW: gl.constexpr = gl.num_warps()
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [NW, 1], [1, 0])
        pid = gl.program_id(0)
        n_tiles = n_rows // ROWS
        my_n = (n_tiles - pid + n_prog - 1) // n_prog
        xs = gl.allocate_shared_memory(x_desc.dtype, [STAGES, ROWS, D], x_desc.layout)
        ys = gl.allocate_shared_memory(y_desc.dtype, [2, ROWS, D], y_desc.layout)
        bars = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
        for s in gl.static_range(STAGES):
            mbarrier.init(bars.index(s), count=1)
        fence_async_shared()
        for s in gl.static_range(STAGES):
            ok = s < my_n
            mbarrier.expect(bars.index(s), x_desc.block_type.nbytes, pred=ok)
            tma.async_copy_global_to_shared(x_desc, [(pid + s * n_prog) * ROWS, 0], bars.index(s), xs.index(s), pred=ok)
        rws = gl.arange(0, ROWS, layout=gl.SliceLayout(1, layout))
        cols = gl.arange(0, D, layout=gl.SliceLayout(0, layout))
        for i in range(my_n):
            s = i % STAGES
            mbarrier.wait(bars.index(s), (i // STAGES) & 1)
            x = xs.index(s).load(layout)
            gl.barrier()  # every warp has read stage s before it is refilled
            nxt = i + STAGES
            ok = nxt < my_n
            mbarrier.expect(bars.index(s), x_desc.block_type.nbytes, pred=ok)
            tma.async_copy_global_to_shared(x_desc, [(pid + nxt * n_prog) * ROWS, 0], bars.index(s), xs.index(s), pred=ok)
            r0 = (pid + i * n_prog) * ROWS
            woffs = gl.expand_dims(((r0 + rws) % H) * D, 1) + gl.expand_dims(cols, 0)
            w = gl.load(w_ptr + woffs)
            inv = gl.rsqrt(gl.sum(x * x, axis=1) / D + eps)
            y = (x * gl.expand_dims(inv, 1)) * w
            b = i % 2
            tma.store_wait(1)  # the store that used buffer b two iterations ago has drained
            ys.index(b).store(y)
            fence_async_shared()
            tma.async_copy_shared_to_global(y_desc, [r0, 0], ys.index(b))
        tma.store_wait(0)
        for s in gl.static_range(STAGES):
            mbarrier.invalidate(bars.index(s))

    for rows, stages in [(8, 4), (16, 4), (32, 3)]:
        lay = NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=32, rank=2)
        dsig = f"tensordesc<fp32[{rows},128],{lay!r}>"
        sig = {"x_desc": dsig, "y_desc": dsig, "w_ptr": "*fp32", "n_rows": "i32", "eps": "fp32", "n_prog": "i32",
               "ROWS": "constexpr", "D": "constexpr", "H": "constexpr", "STAGES": "constexpr"}
        names = g_rms.arg_names
        ce = {(names.index(k),): v for k, v in dict(ROWS=rows, D=D, H=H, STAGES=stages).items()}
        attrs = {(names.index(n),): [["tt.divisibility", 16]] for n in ["w_ptr", "n_rows"]}
        for arch, cc in {"sm_100a": 100, "sm_90a": 90}.items():
            name = f"gluon_tma_ring_R{rows}_S{stages}"
            rec = dict(probe=name, arch=arch)
            try:
                src = GluonASTSource(fn=g_rms, signature={k: sig[k] for k in names}, constexprs=ce, attrs=attrs)
                ck = triton.compile(src, target=GPUTarget("cuda", cc, 32), options=dict(num_warps=4))
                rec.update(sass_summary(ck.asm["cubin"]))
                rec["ptx_hints"] = ptx_hints(ck.asm["ptx"])
                rec["meta_shared"] = ck.metadata.shared
                if arch == "sm_100a":
                    Path(f"{name}.{arch}.ptx").write_text(ck.asm["ptx"])
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:900]}"
            out.append(rec)
            print(json.dumps(rec), flush=True)


out = []
run_triton_probes(out)
if 'with_gluon' in sys.argv:
    try:
        run_gluon_probe(out)
    except Exception:
        traceback.print_exc()
Path("probe2_results.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
