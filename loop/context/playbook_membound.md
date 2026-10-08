# Memory-bound kernel playbook (B200)

For row-wise and elementwise kernels: norms, RoPE, softmax, activations. Examples use #38 (rows of 128 fp32, Q and K
streams). Each design axis lists:

- **when** it helps;
- its **effect on S/M/L** size bands (S ≲ 64 MB, M 100–200 MB, L ≥ 400 MB moved);
- a **Triton 3.7 recipe** (compile-verified for sm_100a unless marked; see `probes/RESULTS.md`);
- CUDA C++ / CuTe DSL pointers;
- its **niche tag(s)**, which the archive uses to keep one best kernel per niche.

"Compile-verified" means it compiled for sm_100a/sm_90a on CSF3 and we inspected the SASS. It does **not** mean it ran.
Every effect estimate here is INFERRED unless labelled.

## 0. Niche tag vocabulary

| Axis | Tags |
|---|---|
| language | `lang:triton`, `lang:gluon`, `lang:tlx`, `lang:cuda`, `lang:cutedsl` |
| memory path | `mem:ldg128`, `mem:ldg256`, `mem:cpasync`, `mem:tma-tensor`, `mem:bulk1d` |
| store path | `st:direct`, `st:bulk` |
| grid | `grid:oneshot`, `grid:persistent`, `grid:persistent-wstat` |
| launches | `launch:fused`, `launch:split`, `launch:pdl` |
| tile | `tile:rows<N>` (N rows per program/CTA iteration), `tile:token` (all heads of one token) |
| reduction | `red:warp`, `red:halfwarp`, `red:cta` |
| cache | `cache:default`, `cache:stream` (inputs/outputs evict_first, reused data evict_last) |
| specialisation | `spec:all`, `spec:S`, `spec:M`, `spec:L`, `dispatch:size` (a dispatcher over specialists) |

The archive niche key is `(mem, grid, spec)`, with the other tags as descriptors (`niche_tags.yaml`). A design card must
give exactly one tag per axis.

## 1. Launch structure — `launch:fused` / `launch:split` / `launch:pdl`

- **Fused** (one kernel covers both Q and K, or every stream of the problem) is the default. CUPTI times first-start to
  last-end, so every extra kernel adds a ramp-up, a tail and a kernel-to-kernel gap. On S-band workloads that is a
  large fraction of a 1.6–7 µs floor. On the L band the loss is under 1%.
- Split only when the two parts need incompatible configurations, and then use PDL so kernel 2's prologue overlaps
  kernel 1's tail:

```python
from triton.language.extra.cuda import gdc_wait, gdc_launch_dependents   # module level, NOT inside @triton.jit
@triton.jit
def k2(...):
    gdc_wait()                 # wait for kernel 1's writes (only needed if k2 reads them)
    ...
    gdc_launch_dependents()    # hint: the next kernel may start
k2[grid](..., launch_pdl=True)   # compile-verified: PTX has griddepcontrol.wait / launch_dependents
```

- CUDA C++: `cudaLaunchKernelEx` with `cudaLaunchAttributeProgrammaticStreamSerialization`, and
  `cudaGridDependencySynchronize()` in the kernel. Only on the current stream.

## 2. Tile shape and rows per program — `tile:rows<N>`, `tile:token`

- Rows per program set the bytes each CTA moves per step, the register use, and the CTA count.
  `tile bytes = ROWS × 512 B` for D=128 fp32.
- For #38, keep ROWS a divisor of 48 (1, 2, 4, 8, 16), so that no masks are needed, each tile's heads form a
  contiguous block, and `tile % (48/ROWS)` identifies the weight block. ROWS = 32 or 64 needs a row mask (the 1×131
  workload).
- S band: small tiles (4–8 rows) give enough CTAs to fill 148 SMs evenly. The smallest workload has 12,288 Q+K rows,
  that is 1,536 CTAs at ROWS=8, about 10.4 per SM in one wave. L band: bigger tiles (16–32 rows) mean fewer CTAs and
  less scheduling and prologue overhead per byte.
- Verified register counts on sm_100a, 4 warps: ROWS=4 → 20 regs, 8 → 32, 16 → 38–40 (40 with cache hints). ROWS=32
  with 8 warps → 38. One ROWS=16 source used 38 regs on sm_100a and 32 on sm_90a, so **always read registers from the
  sm_100a compile**.
- `tile:token` (48 rows = one token) gives a constant weight tile of 24 KB per stream, but 48 is not a power of 2,
  which Triton blocks require. Use 3 × 16-row sub-tiles, or CUDA/Gluon.

## 3. Memory path — `mem:*`

### 3a. `mem:ldg128` — plain vector loads

- **When:** the default, and it runs everywhere (useful for A100/L40S ranking signals). Needs high occupancy so that
  enough loads are outstanding (`b200_arch.md` §3).
- **S:** good (no prologue). **L:** typically about 80% of peak on A100 in the PoC; on B200 it relies on ≥ 48–64 KB per
  SM outstanding.
- **Triton recipe** (one-shot, fused, no masks). The exact kernel below is compile-verified as `playbook_3a_R{4,8,16}`
  (probe7): 32/32/40 regs, no shared memory, `LDG.E.EF.128` / `LDG.E.EL.128` / `STG.E.EF.128`, no `div.full`:

```python
import torch, triton, triton.language as tl
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
    x = tl.load(x_ptr + offs)                                       # LDG.E.128 (no hints on B200: see §6)
    w = tl.load(w_ptr + (rows % H)[:, None] * D + cols[None, :])
    inv = tl.math.rsqrt(tl.sum(x * x, axis=1) * (1.0 / D) + eps)    # * (1/D) is exact; "/" is div.full.f32
    tl.store(y_ptr + offs, (x * inv[:, None]) * w)

def run(query, key, weight_q, weight_k, eps, query_norm, key_norm):
    n_rows = query.numel() // D
    ROWS, WARPS = (8, 4)                     # replace by a size-band table (section 9)
    _qk_rms[(2 * (n_rows // ROWS),)](query, key, weight_q, weight_k, query_norm, key_norm,
                                     n_rows, float(eps), ROWS=ROWS, D=D, H=H, num_warps=WARPS)
```

- Triton picks the layout itself: on sm_100a every load and store above became `LDG/STG.E.128`, 2 per thread for ROWS=8
  and 4 warps.

### 3b. `mem:ldg256` — 256-bit loads/stores (CUDA C++ only)

- **When:** B200-only. It halves the load/store instruction count per byte and doubles the bytes per outstanding
  instruction. That matters more at 36 B/clk/SM (1.5 GHz lock). It cannot be tested on H200, A100 or L40S.
- **Effect:** none expected on S; on L, plausibly a few %. Unmeasured.
- **Triton 3.7 never emits it.** 1-D and 2-D kernels with 8+ contiguous floats per thread still produce `LDG.E.128`
  [VERIFIED-CSF3, probe5].
- **CUDA recipe** (compile-verified, `probes/bulk_probe.cu::k_ldg256` → SASS `LDG.E.NA.ENL2.256`, `STG.E.ENL2.256`;
  needs PTX ISA 8.8, i.e. CUDA ≥ 12.9; the portal has 13.1). A half-warp (16 lanes × 8 floats) covers one 128-float
  row, and the reduction uses `__shfl_xor_sync` with offsets 8, 4, 2, 1 (`red:halfwarp`):

```cuda
asm volatile("ld.global.L1::no_allocate.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
             : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]),"=f"(v[4]),"=f"(v[5]),"=f"(v[6]),"=f"(v[7]) : "l"(p));
asm volatile("st.global.v8.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" :: "l"(q), "f"(v[0]), ..., "f"(v[7]) : "memory");
```

  Addresses must be 32-B aligned. The harness's pointers are 256-B aligned and rows are 512 B, so this holds.

### 3c. `mem:cpasync` — Ampere-style async copies (LDGSTS) through Triton's pipeliner

- **When:** a persistent loop over many tiles per program, where prefetching the next tiles hides latency. It runs on
  A100 too, so A100 gives a relevant signal.
- **S:** neutral or worse (the prologue fills STAGES tiles of a program that has only 1–3 tiles). **L:** can raise
  bytes in flight without high occupancy.
- **Triton recipe** (compile-verified `ptr_persist_R8_S3`: 12 × `LDGSTS.E.BYPASS.128`, 16 KB shared, 32 regs). Use
  `tl.range(..., num_stages=S)`. The kernel argument `num_stages` alone only pipelines loads that feed `tl.dot`:

```python
for t in tl.range(pid, total_tiles, n_prog, num_stages=3):   # n_prog = 148 * PROGS_PER_SM
    ...same body as 3a...
```

### 3d. `mem:tma-tensor` — TMA through tensor descriptors

- **When:** you want many bytes in flight per SM from few instructions; on the L band. Runs on H200 (sm_90a) and B200.
  A100/L40S silently compile a different fallback (sm_80 host-descriptor kernel: 4-byte LDGSTS, 72 regs
  [VERIFIED-CSF3]), so **never use A100/L40S timings for this niche**.
- **S:** risky (descriptor setup and mbarrier prologue). **L:** the most promising Triton path.
- **Host-side descriptors (preferred).** The exact kernel below is compile-verified (`playbook_3d_host`, probe7).
  R8/S3: `UTMALDG.2D` × 6, `UTMASTG.2D` × 2, mbarrier `SYNCS.*`, 36.9 KB shared, 32 regs, no global scratch.
  R16/S4: 46 regs, 106.5 KB shared, so 2 CTAs/SM. The descriptor is encoded on the host for each launch, which is not
  timed. The `run()` host code has not been run:

```python
from triton.tools.tensor_descriptor import TensorDescriptor

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

def run(query, key, weight_q, weight_k, eps, query_norm, key_norm):
    ROWS, STAGES, PROGS_PER_SM = 8, 3, 4
    q2, k2, qo2, ko2 = (t.view(-1, D) for t in (query, key, query_norm, key_norm))
    d = [TensorDescriptor.from_tensor(t, [ROWS, D]) for t in (q2, k2, qo2, ko2)]   # never round_f32_to_tf32
    n_rows = q2.shape[0]
    n_prog = min(n_rows // ROWS, _NUM_SMS * PROGS_PER_SM)   # _NUM_SMS cached at module level
    _qk_rms_tma[(n_prog,)](*d, weight_q, weight_k, n_rows, float(eps), n_prog,
                           ROWS=ROWS, D=D, H=H, STAGES=STAGES, num_warps=4)
```

- **Device-side descriptors** (`tl.make_tensor_descriptor(ptr, shape, strides, block_shape)`) also compile
  (`tma_dev_*`). They need `triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8,
  device="cuda"))` at import, 128 B of global scratch per descriptor per program, and a per-program prologue of
  `tensormap.replace.*`, fences and `CCTL`. Prefer host-side. Triton descriptors support ranks 2–5; a 1-D tensor
  needs a `[n/C, C]` view.
- Triton's pipeliner also pushed the weight `tl.load` through cp.async (12 LDGSTS at R8/S3) and grew shared memory:
  37 KB at R8/S3 and 107 KB at R16/S4, for both host and device variants. 107 KB caps the kernel at 2 CTAs/SM. Check
  `metadata.shared`. Weight-stationary (§4) or Gluon (weights in shared memory once) avoids this.
- Descriptor `.load/.store` take **no cache hints** in Triton 3.7.
- **Gluon** (`triton.experimental.gluon`, ships with fbtriton) gives explicit control of the ring: STAGES, mbarrier
  phases, double-buffered TMA stores, weights loaded once into shared memory. Compile-verified for R8/R16 × S4/S8
  (`probes/probe4.py`; `UTMALDG.2D`, `UTMASTG.2D`, no LDG/STG; 28–40 regs; shared 49 KB (R8,S4), 74 KB (R16,S4),
  107 KB (R16,S8)). Core loop:

```python
for i in range(my_n):                       # tiles of this persistent CTA
    st = i % STAGES
    mbarrier.wait(bars.index(st), (i // STAGES) & 1)
    x = xs.index(st).load(layout)           # smem -> registers (BlockedLayout [1,4],[1,32],[NW,1],[1,0])
    gl.barrier()                            # all warps done with stage st
    ok = i + STAGES < my_n
    mbarrier.expect(bars.index(st), x_desc.block_type.nbytes, pred=ok)
    tma.async_copy_global_to_shared(x_desc, [(pid + (i + STAGES) * n_prog) * ROWS, 0], bars.index(st), xs.index(st), pred=ok)
    w = ws.index((pid + i * n_prog) % G).load(layout)        # weights resident in smem, G = 48 // ROWS
    y = (x * gl.expand_dims(gl.rsqrt(gl.sum(x * x, axis=1) / D + eps), 1)) * w   # verified with "/ D"; prefer * (1.0 / D)
    b = i % 2
    tma.store_wait(1); ys.index(b).store(y); fence_async_shared(); gl.barrier()
    tma.async_copy_shared_to_global(y_desc, [(pid + i * n_prog) * ROWS, 0], ys.index(b))
```

  Gluon pitfall seen on CSF3: `gl.load(ptr_tensor)` returned a tensor without a layout, and the next arithmetic
  failed, so load through TMA/shared memory as above. This may be our misuse. Gluon's host-side
  `TensorDescriptor.from_tensor(t, block, NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=32, rank=2))` is
  not run-tested.
- CuTe DSL: TMA through `cute.nvgpu.cpasync` bulk-tensor copy atoms. Compiling for sm_100a without a GPU works
  (`CUTE_DSL_ARCH=sm_100a` [VERIFIED-CSF3, trivial kernel]); the TMA path itself is not yet verified.

### 3e. `mem:bulk1d` — 1-D bulk copies, no tensor map (CUDA C++)

- **When:** contiguous tiles (a ROWS×128 fp32 tile is one contiguous chunk). It is the simplest TMA-engine path, with
  no descriptor encoding and an L2 cache hint on the copy. Runs on H200 and B200.
- **Recipe** (compile-verified `probes/bulk_probe.cu::k_bulk<ROWS,STAGES>`: `UBLKCP.S.G` loads, `UBLKCP.G.S`
  stores, 25 regs, 25.6 KB (R8,S4) / 50 KB (R16,S4) shared). Producer thread 0:
  `mbarrier.arrive.expect_tx` + `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint`.
  Consumers: `mbarrier.try_wait.parity` loop. One warp per row: a float4 per lane from shared memory, then
  `__shfl_xor_sync` for the sum. Output staged in a double-buffered shared tile, then `fence.proxy.async.shared::cta`
  → `__syncthreads` → `cp.async.bulk.global.shared::cta.bulk_group` + `commit_group`; before reusing a buffer,
  `cp.async.bulk.wait_group.read 1`.
- Above 48 KB use dynamic shared memory: the probe's static 50 KB compiled, but would need the dynamic opt-in to
  launch.
- CuTe DSL equivalent: `cpasync.CopyBulkG2SOp`-style atoms (unverified).

## 4. Grid shape — `grid:oneshot` / `grid:persistent` / `grid:persistent-wstat`

- **One-shot** (one program per tile) has no loop overhead or prologue amortisation issues, and lets the hardware
  scheduler balance. Best for S; fine for M/L with small tiles if occupancy is high. Tail effect: the last partial wave
  is idle bandwidth; it is small when there are many waves.
- **Persistent** (grid = 148 × k, loop over tiles with stride = grid): it amortises prologues (descriptor, barrier
  init, weight load) and enables software pipelining (3c/3d). On S it is risky, because each program has only 1–3
  tiles, so its latency chain is serial. Choose k so that CTAs/SM × k fits occupancy (k = resident CTAs per SM).
- **Weight-stationary persistent** (`grid:persistent-wstat`): program p always handles the same (stream, head group),
  so its weight tile is loaded once, before the loop. Compile-verified `wstat_HB16` (48 regs; 4 weight
  `LDG.E.128` outside the loop, 4 `LDG.E.EF.128` + 4 `STG.E.EF.128` inside):

```python
G2 = 2 * (H // HB)                       # (Q,K) x head groups; n_prog must be a multiple of G2
grp = pid % G2; is_k = grp >= G2 // 2; hg = tl.where(is_k, grp - G2 // 2, grp)
heads = hg * HB + tl.arange(0, HB)
w = tl.load(w_ptr + heads[:, None] * D + cols[None, :])          # once
for tok in range(pid // G2, n_tok, n_prog // G2):
    offs = (tok * H + heads)[:, None] * D + cols[None, :]
    ...
```

  Effect: removes the L1/L2 weight traffic (equal to the input bytes in the naive kernel) and the weight address
  maths. Expected gain is small to moderate on L and unknown on B200.

## 5. Store path — `st:direct` / `st:bulk`

- `st:direct`: `STG.128` (or `.256`) from registers, default cache policy (§6). Simplest. Stores are fire-and-forget, but
  they occupy memory-system queues.
- `st:bulk`: stage in shared memory and issue a TMA/bulk store. Fewer instructions, and it frees registers early.
  Needs `fence.proxy.async` plus a barrier, and costs a shared buffer (double-buffer it).
- For S, direct stores avoid the shared-memory round trip; for L, bulk stores pair naturally with `mem:tma-*`.

## 6. Cache hints — `cache:stream` / `cache:default`

- **On B200: default policy on loads, `evict_last` on output stores** (`cache:stl2`). [portal, 7 October]
  `r5-ldg256-os-r16-stel` (stores `st.global.L2::cache_hint` + `createpolicy.fractional.L2::evict_last`) scored 0.609
  against 0.588 for the same structure with no hints: outputs left in L2 at kernel end are written back after the
  timed window, so M sizes gain 7–9%. `evict_first` on stores costs 0–3% (r2, g2 vs d1). Weights need no hint.
- `evict_first` on the streamed x loads: hurt L by 4% on A100 when combined with evict_last stores (r5 session);
  untested on B200. The Triton mapping of hints is in `b200_arch.md` §7 (`tl.store(..., eviction_policy="evict_last")`
  should give `STG.E.EL`; check with compile_b200). **Do not** use `cache_modifier=".cg"` for loads: it becomes `LDG.E.128.STRONG.GPU`.
  The PoC's EVICT knob was not decisive (A100/L40S). FlashGPU-Sim ignores these hints.

## 7. Reductions — `red:warp` / `red:halfwarp` / `red:cta`

- D=128 fp32 = 512 B = one warp × float4 (`red:warp`: 5 butterfly shuffles), or a half-warp × 8 floats
  (`red:halfwarp`, pairs with 256-bit). Triton lays out a `[ROWS,128]` tile so the row sum is in-warp (10
  `SHFL.BFLY` for 2 rows per warp at ROWS=8, 4 warps). Keep it that way.
- `red:cta` (a row split across warps through shared memory) only pays for long rows (D ≥ 2048). For D=128 it adds
  barriers and is an anti-pattern.
- For long-row problems (for example RMSNorm with hidden size 4096–7168): one CTA (or cluster) per row, a two-level
  reduction (warp shuffle, then shared memory), and the row held in registers between the reduction and the scaling,
  so it is read from DRAM once.

## 8. Small-input tactics — `spec:S`

- **When:** the S band, where t is about 1.5–3× the floor and fixed costs dominate.
- Use one wave: CTAs ≤ 148 × resident CTAs/SM, and spread evenly. Better 1,536 small CTAs than 148 large ones,
  unless each CTA issues all its loads at once.
- Each thread issues all its loads, then reduces, then stores. No loop-carried dependence and no prologue (no device
  descriptors, no shared-memory rings).
- Small CTAs (2–4 warps) with 2–4 rows each launch faster and balance better.
- Effects to measure, not assume: the B200 CTA launch rate and empty-kernel span are unknown (`b200_arch.md` §8).
- Measure S-band candidates on H200 first (closest latency structure), then confirm on B200.

## 9. Per-size dispatch — `dispatch:size`

- Legitimate (choose the config or kernel from shapes only). Pattern:

```python
_BANDS = [  # (max tokens B*S, kernel, meta) -- illustrative; fill from measurements
    (600,   _qk_rms,     dict(ROWS=4,  num_warps=2)),   # S band: <= 2*293 tokens
    (2048,  _qk_rms,     dict(ROWS=8,  num_warps=4)),   # M band
    (1<<62, _qk_rms_tma, dict(ROWS=16, STAGES=4)),      # L band
]
def run(query, key, weight_q, weight_k, eps, query_norm, key_norm):
    tokens = query.shape[0] * query.shape[1]
    kern, meta = next((k, m) for lim, k, m in _BANDS if tokens <= lim)
    ...launch exactly one kernel...
```

- Every specialist gets compiled during correctness round 0 of the first workload that uses it, inside the 300 s limit.
  Triton compiles take about 1–3 s each, CuTe DSL about 20 s [VERIFIED-CSF3 for a trivial kernel]. Keep it to ≤ 4–5
  specialists.
- In the PoC, picking the best of 72 knob variants per size gained only 1.4–2.3%. Dispatch pays off only between
  **structurally different** specialists (for example `spec:S` pointer one-shot against `spec:L` TMA persistent).

## 10. Register pressure and occupancy (all designs)

- Read regs and shared memory from the **sm_100a** compile (`kernel.n_regs`, `metadata.shared`, or `cuobjdump
  -res-usage`). ≤ 32 regs keeps 2048 threads per SM. Spills (`LOCAL > 0`) are a red flag for streaming kernels.
- Triton `num_warps` 1–8. Fewer warps per CTA means more CTAs per SM, which helps S (the PoC's best L40S variant used
  1 warp; on A100, 2 warps with ROWS=32 persistent).
- Bytes in flight per SM = resident CTAs × outstanding bytes per CTA. Target ≥ 48–64 KB on B200, and write the
  estimate in the design card.

## 11. Anti-patterns

VERIFIED-CSF3 where marked; the rest are INFERRED from the harness code or the B200 numbers.

| Anti-pattern | Why |
|---|---|
| Two launches (Q then K), or any torch op in `run()` | timed gaps and extra ramp-ups; torch fill/copy kernels can be confused with harness setup in CUPTI matching |
| `tl.range(..., warp_specialize=True)` on non-matmul loops | sm_100a compile **crashes** (`TritonGPULoadMMASpecialization`); sm_90a ignores it [VERIFIED-CSF3] |
| Expecting Triton to emit 256-bit LDG | it does not [VERIFIED-CSF3]; use CUDA C++ |
| `cache_modifier=".cg"` on streaming loads | becomes `LDG...STRONG.GPU` [VERIFIED-CSF3] |
| Device-side descriptors in the S path | per-program tensormap build, fences and global scratch |
| Shared-memory rings so large that only 1 CTA fits per SM, in a one-shot grid | kills occupancy and in-flight bytes |
| Persistent grid on S-band workloads | 1–3 tiles per program, so serial latency |
| Persistent grid sized from another chip's occupancy | the extra programs wait for whole programs to finish, then run a full share: a second wave, up to about 1.5× slower. PoC v028 (ROWS=32, 2 warps, 8 programs per SM) uses 128 registers on sm_90a (8 fit) but **149 on sm_100a (6 fit)**. It ran 36.4 µs on H200 and **35.9 µs on B200** (score 0.451), so it gained nothing from 1.7× more bandwidth. Size `k` from the **sm_100a** register and shared-memory counts, or read `n_regs` from the compiled kernel at runtime [VERIFIED-CSF3; portal result 6 Oct, cause INFERRED] |
| Fp32 `/` for constant divisors | Triton emits `div.full.f32` (approximate, ≤ 2 ulp; Gluon showed an extra `MUFU.RCP`) [VERIFIED-CSF3]; multiply by the reciprocal |
| Masks on every element when ROWS ∤ 48 | extra predicates; choose ROWS dividing the row multiple |
| `@triton.autotune` with large config lists | compile/benchmark time inside the 300 s limit, noisy choice; use a band table |
| int32 offsets on tensors ≥ 2^31 elements (other problems) | overflow; cast row index to int64 |
| Any cache keyed on `data_ptr`, data values or call count | forbidden (reward hack) |
| Reusing a TMA descriptor across calls | pointers move every call: wrong results |

## 12. Porting notes

- **Triton → CUDA C++**: the binding is in `harness_scoring.md` §2. The kernel is the same algorithm. Gains available
  only in CUDA: 256-bit accesses, the 1-D bulk path, explicit `createpolicy` on bulk copies, exact control of the
  prologue. Use `__launch_bounds__(threads, minBlocks)` to pin registers. Launch on
  `at::cuda::getCurrentCUDAStream()`; check the `is_contiguous()` and alignment assumptions on the host.
- **Triton → Gluon**: the same Python harness interface. You control layouts, rings and barriers. Start from
  `probes/probe4.py`.
- **Any → CuTe DSL**: compile once (`cute.compile` at first call), `mark_layout_dynamic()` so all sizes share one
  binary, and `from_dlpack` for the inputs each call. Allow about 20 s compile time per kernel.

## 13. Other memory-bound row-wise problems

The same axes apply. Recompute: bytes per workload (inputs read once plus outputs written once; count broadcast
operands once), the floor per workload at 8 TB/s, the S/M/L bands, the row length (D) for the reduction pattern, and
which tensors are reused (weights, cos/sin tables → `evict_last`, or stationary). Then copy the structure of
the problem card (`problems/<name>/card.md`) into a new problem card.
