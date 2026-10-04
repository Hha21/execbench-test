"""Compile-only probes for the context docs (no GPU). Triton 3.7 (fbtriton 3.7.1).

For each probe kernel and target arch: registers, shared memory, spills, and a SASS opcode
summary (global loads/stores and their widths/hints, TMA, barriers), plus PTX hint lines.
"""
import collections, json, re, subprocess, sys, tempfile, traceback
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

TOOLS = Path(triton.__file__).parent / "backends" / "nvidia" / "bin"
D = 128
H = 48


# ---------------------------------------------------------------- probe kernels
@triton.jit
def k_ptr(q_ptr, k_ptr, wq_ptr, wk_ptr, qo_ptr, ko_ptr, n_rows, eps,
          ROWS: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
          LOAD_EVICT: tl.constexpr, STORE_EVICT: tl.constexpr, LOAD_CM: tl.constexpr, STORE_CM: tl.constexpr):
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(n_rows, ROWS)
    is_k = pid >= n_tiles
    tile = tl.where(is_k, pid - n_tiles, pid)
    x_ptr = tl.where(is_k, k_ptr, q_ptr)
    w_ptr = tl.where(is_k, wk_ptr, wq_ptr)
    y_ptr = tl.where(is_k, ko_ptr, qo_ptr)
    rows = tile * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, D)
    mask = (rows < n_rows)[:, None]
    offs = rows[:, None] * D + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0, eviction_policy=LOAD_EVICT, cache_modifier=LOAD_CM)
    w = tl.load(w_ptr + (rows % H)[:, None] * D + cols[None, :], mask=mask, other=0.0, eviction_policy="evict_last")
    inv = tl.math.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    y = (x * inv[:, None]) * w
    tl.store(y_ptr + offs, y, mask=mask, eviction_policy=STORE_EVICT, cache_modifier=STORE_CM)


@triton.jit
def k_ptr_nomask(q_ptr, k_ptr, wq_ptr, wk_ptr, qo_ptr, ko_ptr, n_rows, eps,
                 ROWS: tl.constexpr, D: tl.constexpr, H: tl.constexpr):
    # n_rows = B*S*48 is a multiple of 48; with ROWS | 48 (1,2,4,8,16) no row mask is needed.
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
    x = tl.load(x_ptr + offs)
    w = tl.load(w_ptr + (rows % H)[:, None] * D + cols[None, :])
    inv = tl.math.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    tl.store(y_ptr + offs, (x * inv[:, None]) * w)


@triton.jit
def k_ptr_persist(q_ptr, k_ptr, wq_ptr, wk_ptr, qo_ptr, ko_ptr, n_rows, eps, n_prog,
                  ROWS: tl.constexpr, D: tl.constexpr, H: tl.constexpr, STAGES: tl.constexpr):
    pid = tl.program_id(0)
    n_tiles = n_rows // ROWS
    total = 2 * n_tiles
    cols = tl.arange(0, D)
    for t in tl.range(pid, total, n_prog, num_stages=STAGES):
        is_k = t >= n_tiles
        tile = tl.where(is_k, t - n_tiles, t)
        x_ptr = tl.where(is_k, k_ptr, q_ptr)
        w_ptr = tl.where(is_k, wk_ptr, wq_ptr)
        y_ptr = tl.where(is_k, ko_ptr, qo_ptr)
        rows = tile * ROWS + tl.arange(0, ROWS)
        offs = rows[:, None] * D + cols[None, :]
        x = tl.load(x_ptr + offs)
        w = tl.load(w_ptr + (rows % H)[:, None] * D + cols[None, :])
        inv = tl.math.rsqrt(tl.sum(x * x, axis=1) / D + eps)
        tl.store(y_ptr + offs, (x * inv[:, None]) * w)


@triton.jit
def k_tma_dev(q_ptr, k_ptr, wq_ptr, wk_ptr, qo_ptr, ko_ptr, n_rows, eps, n_prog,
              ROWS: tl.constexpr, D: tl.constexpr, H: tl.constexpr, STAGES: tl.constexpr):
    # Device-side descriptors (need triton.set_allocator at runtime). Persistent grid over Q tiles then K tiles.
    pid = tl.program_id(0)
    qd = tl.make_tensor_descriptor(q_ptr, shape=[n_rows, D], strides=[D, 1], block_shape=[ROWS, D])
    kd = tl.make_tensor_descriptor(k_ptr, shape=[n_rows, D], strides=[D, 1], block_shape=[ROWS, D])
    qod = tl.make_tensor_descriptor(qo_ptr, shape=[n_rows, D], strides=[D, 1], block_shape=[ROWS, D])
    kod = tl.make_tensor_descriptor(ko_ptr, shape=[n_rows, D], strides=[D, 1], block_shape=[ROWS, D])
    n_tiles = n_rows // ROWS
    cols = tl.arange(0, D)
    for t in tl.range(pid, n_tiles, n_prog, num_stages=STAGES):
        r0 = t * ROWS
        rows = r0 + tl.arange(0, ROWS)
        hq = (rows % H)[:, None] * D + cols[None, :]
        x = qd.load([r0, 0])
        inv = tl.math.rsqrt(tl.sum(x * x, axis=1) / D + eps)
        qod.store([r0, 0], (x * inv[:, None]) * tl.load(wq_ptr + hq))
        x = kd.load([r0, 0])
        inv = tl.math.rsqrt(tl.sum(x * x, axis=1) / D + eps)
        kod.store([r0, 0], (x * inv[:, None]) * tl.load(wk_ptr + hq))


@triton.jit
def k_tma_host(qd, kd, qod, kod, wq_ptr, wk_ptr, n_rows, eps, n_prog,
               ROWS: tl.constexpr, D: tl.constexpr, H: tl.constexpr, STAGES: tl.constexpr):
    # Host-side descriptors (triton.tools.tensor_descriptor.TensorDescriptor), no allocator needed.
    pid = tl.program_id(0)
    n_tiles = n_rows // ROWS
    cols = tl.arange(0, D)
    for t in tl.range(pid, n_tiles, n_prog, num_stages=STAGES):
        r0 = t * ROWS
        rows = r0 + tl.arange(0, ROWS)
        hq = (rows % H)[:, None] * D + cols[None, :]
        x = qd.load([r0, 0])
        inv = tl.math.rsqrt(tl.sum(x * x, axis=1) / D + eps)
        qod.store([r0, 0], (x * inv[:, None]) * tl.load(wq_ptr + hq))
        x = kd.load([r0, 0])
        inv = tl.math.rsqrt(tl.sum(x * x, axis=1) / D + eps)
        kod.store([r0, 0], (x * inv[:, None]) * tl.load(wk_ptr + hq))


@triton.jit
def k_gdc(q_ptr, qo_ptr, n_rows, eps, ROWS: tl.constexpr, D: tl.constexpr):
    # PDL hooks: wait for the previous grid's memory, then let the next grid start early.
    from triton.language.extra.cuda import gdc_wait, gdc_launch_dependents
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, D)
    offs = rows[:, None] * D + cols[None, :]
    gdc_wait()
    x = tl.load(q_ptr + offs)
    gdc_launch_dependents()
    inv = tl.math.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    tl.store(qo_ptr + offs, x * inv[:, None])


# ---------------------------------------------------------------- compile + inspect
def compile_one(fn, sig, constexprs, cc, opts, ptr_names=()):
    names = fn.arg_names
    ce = {(names.index(n),): v for n, v in constexprs.items()}
    full_sig = {n: sig.get(n, "constexpr") for n in names}
    attrs = {(names.index(n),): [["tt.divisibility", 16]] for n in ptr_names}
    src = ASTSource(fn=fn, signature=full_sig, constexprs=ce, attrs=attrs)
    return triton.compile(src, target=GPUTarget("cuda", cc, 32), options=opts)


def sass_summary(cubin):
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(cubin); f.flush()
        res = subprocess.run([TOOLS / "cuobjdump", "-res-usage", f.name], capture_output=True, text=True).stdout
        sass = subprocess.run([TOOLS / "cuobjdump", "-sass", f.name], capture_output=True, text=True).stdout
    m = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", res)
    ops = collections.Counter()
    for line in sass.splitlines():
        mm = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_.]+)", line)
        if mm:
            ops[mm[1]] += 1
    keep = {k: v for k, v in ops.items() if re.match(r"(LDG|STG|LDS|STS|UTMA|UBLKCP|UTMACCTL|SYNCS|LDGSTS|BAR|MUFU|SHFL|CCTL|FENCE|MEMBAR|ULDC|LDC)", k)}
    return dict(regs=int(m[1]), stack=int(m[2]), shared=int(m[3]), local=int(m[4]), ops=dict(sorted(keep.items())),
                n_sass=sum(ops.values()))


def ptx_hints(ptx):
    lines = [l.strip() for l in ptx.splitlines()
             if re.search(r"createpolicy|L2::cache_hint|evict_|\.cs\.|\.cg\.|\.lu\.|cp\.async|mbarrier|griddepcontrol|tensormap|prefetch", l)]
    c = collections.Counter(re.sub(r"[%\w]+_\d+|\[.*?\]|%\w+|\d+", "_", l)[:90] for l in lines)
    return dict(c.most_common(12))


PTRS6 = ["q_ptr", "k_ptr", "wq_ptr", "wk_ptr", "qo_ptr", "ko_ptr"]
SIG6 = {n: "*fp32" for n in PTRS6} | {"n_rows": "i32", "eps": "fp32", "n_prog": "i32"}

PROBES = []
for rows, warps in [(8, 4), (4, 4), (16, 4), (32, 8)]:
    PROBES.append((f"ptr_R{rows}_W{warps}", k_ptr, SIG6,
                   dict(ROWS=rows, D=D, H=H, LOAD_EVICT="", STORE_EVICT="", LOAD_CM="", STORE_CM=""), dict(num_warps=warps, num_stages=1), PTRS6 + ["n_rows"]))
PROBES += [
    ("ptr_evict_first", k_ptr, SIG6, dict(ROWS=8, D=D, H=H, LOAD_EVICT="evict_first", STORE_EVICT="evict_first", LOAD_CM="", STORE_CM=""), dict(num_warps=4, num_stages=1), PTRS6 + ["n_rows"]),
    ("ptr_cg_cs", k_ptr, SIG6, dict(ROWS=8, D=D, H=H, LOAD_EVICT="", STORE_EVICT="", LOAD_CM=".cg", STORE_CM=".cs"), dict(num_warps=4, num_stages=1), PTRS6 + ["n_rows"]),
    ("ptr_nomask_R8", k_ptr_nomask, SIG6, dict(ROWS=8, D=D, H=H), dict(num_warps=4, num_stages=1), PTRS6 + ["n_rows"]),
    ("ptr_persist_R8_S1", k_ptr_persist, SIG6, dict(ROWS=8, D=D, H=H, STAGES=1), dict(num_warps=4), PTRS6 + ["n_rows"]),
    ("ptr_persist_R8_S3", k_ptr_persist, SIG6, dict(ROWS=8, D=D, H=H, STAGES=3), dict(num_warps=4), PTRS6 + ["n_rows"]),
    ("tma_dev_R8_S3", k_tma_dev, SIG6, dict(ROWS=8, D=D, H=H, STAGES=3), dict(num_warps=4), PTRS6 + ["n_rows"]),
    ("tma_dev_R16_S4", k_tma_dev, SIG6, dict(ROWS=16, D=D, H=H, STAGES=4), dict(num_warps=4), PTRS6 + ["n_rows"]),
    ("tma_host_R8_S3", k_tma_host, {"qd": "tensordesc<fp32[8,128]>", "kd": "tensordesc<fp32[8,128]>", "qod": "tensordesc<fp32[8,128]>", "kod": "tensordesc<fp32[8,128]>",
                                    "wq_ptr": "*fp32", "wk_ptr": "*fp32", "n_rows": "i32", "eps": "fp32", "n_prog": "i32"},
     dict(ROWS=8, D=D, H=H, STAGES=3), dict(num_warps=4), ["wq_ptr", "wk_ptr", "n_rows"]),
    ("gdc_pdl", k_gdc, {"q_ptr": "*fp32", "qo_ptr": "*fp32", "n_rows": "i32", "eps": "fp32"}, dict(ROWS=8, D=D), dict(num_warps=4, launch_pdl=True), ["q_ptr", "qo_ptr"]),
]

def main():
    ARCHS = {"sm_100a": 100, "sm_90a": 90, "sm_80": 80}
    only = sys.argv[1:]
    out = []
    for name, fn, sig, ce, opts, ptrs in PROBES:
        if only and name not in only:
            continue
        for arch, cc in ARCHS.items():
            rec = dict(probe=name, arch=arch)
            try:
                ck = compile_one(fn, sig, ce, cc, opts, ptrs)
                rec.update(sass_summary(ck.asm["cubin"]))
                rec["ptx_hints"] = ptx_hints(ck.asm["ptx"])
                rec["meta_shared"] = ck.metadata.shared
                gs = getattr(ck.metadata, "global_scratch_size", None)
                if gs is not None:
                    rec["global_scratch"] = gs
                if arch == "sm_100a":
                    Path(f"{name}.{arch}.ptx").write_text(ck.asm["ptx"])
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            out.append(rec)
            print(json.dumps(rec), flush=True)
    Path("probe1_results.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
    print("triton", triton.__version__)


if __name__ == "__main__":
    main()
