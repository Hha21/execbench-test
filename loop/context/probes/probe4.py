"""Gluon explicit TMA ring for per-head RMSNorm (one tensor per launch), compile-only."""
import json, sys
from pathlib import Path
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared
from triton.experimental.gluon._runtime import GluonASTSource
from triton.experimental.gluon.language._layouts import NVMMASharedLayout
sys.path.insert(0, ".")
from probe1 import sass_summary, ptx_hints


@gluon.jit
def g_rms(x_desc, y_desc, w_desc, n_rows, eps, n_prog,
          ROWS: gl.constexpr, D: gl.constexpr, H: gl.constexpr, STAGES: gl.constexpr):
    NW: gl.constexpr = gl.num_warps()
    G: gl.constexpr = H // ROWS                      # ROWS must divide 48: tile t uses weight block t % G
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [NW, 1], [1, 0])
    pid = gl.program_id(0)
    n_tiles = n_rows // ROWS
    my_n = (n_tiles - pid + n_prog - 1) // n_prog
    xs = gl.allocate_shared_memory(x_desc.dtype, [STAGES, ROWS, D], x_desc.layout)
    ys = gl.allocate_shared_memory(y_desc.dtype, [2, ROWS, D], y_desc.layout)
    ws = gl.allocate_shared_memory(w_desc.dtype, [G, ROWS, D], w_desc.layout)
    bars = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    wbar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    for s in gl.static_range(STAGES):
        mbarrier.init(bars.index(s), count=1)
    mbarrier.init(wbar, count=1)
    fence_async_shared()
    # weights: G boxes of [ROWS, D], loaded once per CTA (24 KB per tensor)
    mbarrier.expect(wbar, G * w_desc.block_type.nbytes)
    for g in gl.static_range(G):
        tma.async_copy_global_to_shared(w_desc, [g * ROWS, 0], wbar, ws.index(g))
    # prologue: fill the ring
    for s in gl.static_range(STAGES):
        ok = s < my_n
        mbarrier.expect(bars.index(s), x_desc.block_type.nbytes, pred=ok)
        tma.async_copy_global_to_shared(x_desc, [(pid + s * n_prog) * ROWS, 0], bars.index(s), xs.index(s), pred=ok)
    mbarrier.wait(wbar, 0)
    for i in range(my_n):
        st = i % STAGES
        t = pid + i * n_prog
        mbarrier.wait(bars.index(st), (i // STAGES) & 1)
        x = xs.index(st).load(layout)
        gl.barrier()                                   # all warps have read stage st
        nxt = i + STAGES
        ok = nxt < my_n
        mbarrier.expect(bars.index(st), x_desc.block_type.nbytes, pred=ok)
        tma.async_copy_global_to_shared(x_desc, [(pid + nxt * n_prog) * ROWS, 0], bars.index(st), xs.index(st), pred=ok)
        w = ws.index(t % G).load(layout)
        inv = gl.rsqrt(gl.sum(x * x, axis=1) / D + eps)
        y = (x * gl.expand_dims(inv, 1)) * w
        b = i % 2
        tma.store_wait(1)                              # store issued from buffer b two tiles ago has drained
        ys.index(b).store(y)
        fence_async_shared()
        gl.barrier()
        tma.async_copy_shared_to_global(y_desc, [t * ROWS, 0], ys.index(b))
    tma.store_wait(0)
    for s in gl.static_range(STAGES):
        mbarrier.invalidate(bars.index(s))
    mbarrier.invalidate(wbar)


out = []
for rows, stages, warps in [(8, 4, 4), (16, 4, 4), (16, 8, 4), (16, 4, 8)]:
    lay = NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=32, rank=2)
    dsig = f"tensordesc<fp32[{rows},128],{lay!r}>"
    sig = {"x_desc": dsig, "y_desc": dsig, "w_desc": dsig, "n_rows": "i32", "eps": "fp32", "n_prog": "i32",
           "ROWS": "constexpr", "D": "constexpr", "H": "constexpr", "STAGES": "constexpr"}
    names = g_rms.arg_names
    ce = {(names.index(k),): v for k, v in dict(ROWS=rows, D=128, H=48, STAGES=stages).items()}
    attrs = {(names.index("n_rows"),): [["tt.divisibility", 16]]}
    for arch, cc in {"sm_100a": 100, "sm_90a": 90}.items():
        name = f"gluon_ring_R{rows}_S{stages}_W{warps}"
        rec = dict(probe=name, arch=arch)
        try:
            src = GluonASTSource(fn=g_rms, signature={k: sig[k] for k in names}, constexprs=ce, attrs=attrs)
            ck = triton.compile(src, target=GPUTarget("cuda", cc, 32), options=dict(num_warps=warps))
            rec.update(sass_summary(ck.asm["cubin"]))
            rec["ptx_hints"] = ptx_hints(ck.asm["ptx"])
            rec["meta_shared"] = ck.metadata.shared
            if arch == "sm_100a":
                Path(f"{name}.{arch}.ptx").write_text(ck.asm["ptx"])
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[-700:]}"
        out.append(rec)
        print(json.dumps(rec), flush=True)
Path("probe4_results.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
