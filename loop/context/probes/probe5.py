import json, sys
import triton, triton.language as tl
sys.path.insert(0, ".")
from probe1 import sass_summary, compile_one

@triton.jit
def k_1d(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(y_ptr + offs, x * 2.0)

@triton.jit
def k_rows(x_ptr, y_ptr, ROWS: tl.constexpr, D: tl.constexpr):
    # 2D tile with an explicit 8-wide contiguous chunk per thread via reshape
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.arange(0, D)
    offs = r[:, None] * D + c[None, :]
    x = tl.load(x_ptr + offs)
    inv = tl.math.rsqrt(tl.sum(x * x, axis=1) * (1.0 / D) + 1e-6)
    tl.store(y_ptr + offs, x * inv[:, None])

for name, fn, ce, opts in [("1d_B1024_W4", k_1d, dict(BLOCK=1024), dict(num_warps=4)),
                           ("1d_B2048_W4", k_1d, dict(BLOCK=2048), dict(num_warps=4)),
                           ("rows_R16_W2", k_rows, dict(ROWS=16, D=128), dict(num_warps=2)),
                           ("rows_R8_W1", k_rows, dict(ROWS=8, D=128), dict(num_warps=1))]:
    sig = {"x_ptr": "*fp32", "y_ptr": "*fp32", "n": "i32"}
    for arch, cc in {"sm_100a": 100, "sm_90a": 90}.items():
        try:
            ck = compile_one(fn, sig, ce, cc, opts, ["x_ptr", "y_ptr"])
            r = sass_summary(ck.asm["cubin"])
            print(name, arch, r["regs"], {k: v for k, v in r["ops"].items() if k.startswith(("LDG", "STG"))})
        except Exception as e:
            print(name, arch, "ERR", str(e)[:200])
