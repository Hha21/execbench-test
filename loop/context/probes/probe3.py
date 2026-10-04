import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource

@gluon.jit
def g(x_ptr, w_ptr, y_ptr, eps, VARIANT: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    rws = gl.arange(0, 8, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    offs = gl.expand_dims(rws * 128, 1) + gl.expand_dims(cols, 0)
    p = x_ptr + offs
    gl.static_print(offs.type, p.type)
    x = gl.load(p)
    gl.static_print(x.type)
    w = gl.load(w_ptr + offs)
    s = gl.sum(x * x, axis=1)
    if VARIANT == 0:
        inv = gl.rsqrt(s / 128 + eps)
    elif VARIANT == 1:
        inv = gl.rsqrt(s)
    else:
        inv = s
    gl.static_print(s, inv)
    y = (x * gl.expand_dims(inv, 1)) * w
    gl.store(y_ptr + offs, y)

for v in (0,):
    names = g.arg_names
    sig = {"x_ptr": "*fp32", "w_ptr": "*fp32", "y_ptr": "*fp32", "eps": "fp32", "VARIANT": "constexpr"}
    try:
        src = GluonASTSource(fn=g, signature=sig, constexprs={(names.index("VARIANT"),): v})
        ck = triton.compile(src, target=GPUTarget("cuda", 100, 32), options=dict(num_warps=4))
        print(v, "OK")
    except Exception as e:
        print(v, "ERR", str(e)[-300:])
