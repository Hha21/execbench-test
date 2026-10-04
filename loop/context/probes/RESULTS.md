# Compile-only probes on CSF3 (2026-10-04)

These scripts produced every VERIFIED-CSF3 compiler fact in the context docs. They compile only and never run a kernel;
there is no GPU on the login node.

How they were run (seconds of CPU on the CSF3 login node, files in `/scratch/t95317ha/solx/tmp/ctxdocs/`):

```bash
ssh csf3 'source /scratch/t95317ha/solx/env.sh; cd /scratch/t95317ha/solx/tmp/ctxdocs && \
  TRITON_CACHE_DIR=$PWD/tcache python probe1.py'          # also probe2/4/5/7; probe6 = CuTe DSL
# CUDA C++: nvcc 12.8 makes PTX, ptxas-blackwell 13.1 (shipped with fbtriton) assembles it, because v8 loads need PTX 8.8
module load libs/cuda/12.8.1; nvcc -arch=sm_100a -ptx -O3 --use_fast_math -std=c++17 bulk_probe.cu -o bulk_probe.ptx
sed -i 's/^.version 8.7/.version 8.8/' bulk_probe.ptx
$TRITON_NV_BIN/ptxas-blackwell -arch=sm_100a -O3 bulk_probe.ptx -o bulk_probe.cubin; cuobjdump -res-usage -sass bulk_probe.cubin
```

Environment: fbtriton 3.7.1 (`triton.__version__` 3.7.0+fb.beta), torch 2.9.0+cu130, nvidia-cutlass-dsl 4.4.2,
Triton's `ptxas-blackwell` 13.1.80 (used for arch ≥ 100), `ptxas` 12.9.86 (otherwise). Triton targets use
`GPUTarget("cuda", 100|90|80, 32)`; pointers and `n_rows` carry the `tt.divisibility=16` attribute, as at runtime.

## Results (sm_100a unless noted)

| Probe | What | Regs | Shared | Key SASS | Finding |
|---|---|---|---|---|---|
| probe1 `ptr_R{4,8,16}_W4`, `ptr_R32_W8` | PoC-style pointer tile with row mask | 20 / 32 / 38 / 38 | 0 | `LDG.E.128`, `STG.E.128`, `MUFU.RSQ`, `SHFL.BFLY` | the row mask keeps 128-bit vectors; sm_90a used 32 regs for R16 where sm_100a used 38 |
| probe1 `ptr_evict_first` | `eviction_policy="evict_first"` on load and store | 32 | 0 | `LDG.E.EF.128`, `STG.E.EF.128` (`evict_last` → `LDG.E.EL.128`) | hints reach the SASS |
| probe1 `ptr_cg_cs` | `cache_modifier=".cg"` load, `".cs"` store | 32 | 0 | `LDG.E.128.STRONG.GPU`, `STG.E.EF.128` | `.cg` gives a strong (coherent) load |
| probe1 `ptr_persist_R8_S1/S3` | persistent loop, `tl.range(num_stages=1/3)` | 32 | 0 / 16 KB (meta) | S3: 12 × `LDGSTS.E.BYPASS.128`, `LDGDEPBAR` | `tl.range(num_stages)` pipelines pointer loads through cp.async |
| probe1 `tma_dev_R8_S3`, `tma_dev_R16_S4` | device-side `tl.make_tensor_descriptor` | 32 / 40 | 36.9 KB / 106.6 KB | `UTMALDG.2D`, `UTMASTG.2D`, `SYNCS.*`, `UTMACCTL.IV`, `CCTL...`, `tensormap.replace` ×N in PTX | global scratch 512 B per program (4 descriptors); weights also pipelined through LDGSTS |
| probe1 `tma_host_R8_S3` | host-side `tensordesc<fp32[8,128]>` arguments | 32 | 36.9 KB | `UTMALDG.2D` ×6, `UTMASTG.2D` ×2, no LDG/STG for Q/K | no scratch, no tensormap prologue. **sm_80 fallback**: 72 regs, 4-byte `LDGSTS.E`, `STG.E` (different code) |
| probe2 `gdc_pdl` | `gdc_wait` / `gdc_launch_dependents`, `launch_pdl=True` | 22 | 0 | PTX `griddepcontrol.wait/launch_dependents`; `metadata.launch_pdl=True` | PDL works; import at module level (an import inside `@triton.jit` fails) |
| probe2 `tma_ws_R8_S3` | `tl.range(..., warp_specialize=True)` on a TMA RMS loop | — | — | — | **sm_100a: compiler crash** (`TritonGPULoadMMASpecialization`); sm_90a: compiles, no specialisation |
| probe2 `wstat_HB16` | weight-stationary persistent (fixed head group per program) | 48 | 0 | weight `LDG.E.128` ×4 outside the loop; `LDG.E.EF.128` ×4, `STG.E.EF.128` ×4 inside | works |
| probe3 | Gluon `gl.load(ptr_tensor)` | — | — | — | the result had no layout, so `x*x` failed (possibly our misuse) |
| probe4 `gluon_ring_R{8,16}_S{4,8}_W{4,8}` | explicit Gluon TMA ring, weights in smem, double-buffered TMA store | 28–40 | 49.2 KB (R8S4), 73.8 KB (R16S4), 106.6 KB (R16S8) | `UTMALDG.2D`, `UTMASTG.2D`, `SYNCS.*`, no LDG/STG; PTX shows `mul.f32x2` | works on sm_100a and sm_90a |
| probe5 | 1-D kernel (8–16 floats per thread) and 2-D row tiles | 18–56 | 0 | only `LDG.E.128`/`STG.E.128` | **Triton 3.7 never emits 256-bit accesses** |
| probe6 | CuTe DSL `cute.compile` of a trivial kernel, `CUTE_DSL_ARCH=sm_100a`, CPU tensors | — | — | — | compiles without a GPU in about 21 s (including import) |
| probe7 `playbook_3a_R{4,8,16}` | exact §3a playbook kernel | 32 / 32 / 40 | 0 | `LDG.E.EF.128`, `LDG.E.EL.128`, `STG.E.EF.128`; no `div.full` | snippet verified |
| probe7 `playbook_3d_host_R8_S3`, `_R16_S4` | exact §3d playbook kernel | 32 / 46 | 36.9 KB / 106.5 KB | `UTMALDG.2D`, `UTMASTG.2D` | snippet verified; R16S4 allows only 2 CTAs/SM |
| bulk_probe `k_ldg256` | CUDA C++ `ld/st.global.v8.f32` (half-warp per row) | 32 | 0 | `LDG.E.NA.ENL2.256` ×4, `STG.E.ENL2.256` ×4 | 256-bit works from CUDA C++ (needs PTX 8.8 / CUDA ≥ 12.9) |
| bulk_probe `k_bulk<8,4>`, `<16,4>` | CUDA C++ 1-D `cp.async.bulk` ring + bulk store | 25 / 25 | 25.6 KB / 50.2 KB (static) | `UBLKCP.S.G` ×8, `UBLKCP.G.S`, `SYNCS.*`, `ELECT` | works; above 48 KB use dynamic shared memory |

Other facts from the same session:

- Triton's `tl.math.rsqrt` → PTX `rsqrt.approx.ftz.f32` → `MUFU.RSQ`. Fp32 `/` → `div.full.f32`.
- fbtriton ships a ctypes "no-compile launcher" (`TRITON_USE_NO_COMPILE_LAUNCHER`) and `triton/backends/nvidia/launch.h`
  on CSF3 is our symlink to `../../runtime/launch.h`.
- Not verified: anything at runtime (correctness, timing, `set_allocator`, host descriptor encoding, PDL overlap),
  CuTe DSL TMA atoms, TLX.

## Re-run when

The portal's toolchain changes (the Dockerfile pins), or a design relies on a compiler behaviour not listed here.
Add the probe here and cite it as VERIFIED-CSF3 with the probe name.
