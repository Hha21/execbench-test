# Design session 20261007-222420_01-structural_mutation-ml

Task: {"operation": "structural_mutation", "parents": ["r6-s-ldef-nc-disp"], "band": "M,L", "gain": 0.01577762095760471, "instructions": "Goal: cut medium and large input latency (B*S > 600) on B200. The parent reaches 6.47 TB/s (M) and 6.94 TB/s (L) effective; a further 5% would add about +0.016 to the score. Its fit is 7.6 TB/s, above the physical read+write rate: outputs still in L2 when the kernel ends are written back after the timed window (see problem_038.md section 7), so think about what occupies L2 at the end, not only bandwidth. Propose the single change most likely to help, correct for every shape, one launch. Slower on B200 than the parent in these bands already: d1-os-r8w4-nohint (ldg128/oneshot): +7%; g2-os-r16w8 (ldg128/oneshot): +9%; g2-os-r8w4 (ldg128/oneshot): +9%; g2-tma-r8s3 (tma-tensor/persistent): +27%; g2-tmaws-hb8s4 (tma-tensor/persistent-wstat): +20%; g2-ws-hb16w8s3 (cpasync/persistent-wstat): +24%; r2-ldg256-os-r16 (ldg256/oneshot): +9%; r3-cute-ldg256-os-r16 (ldg256/oneshot): +5%; v028 (ldg128/persistent): +59%; v039 (ldg128/persistent): +22%; v040 (ldg128/oneshot): +144%; v050 (ldg128/oneshot): +38%. Ideas already proposed but not yet measured on B200 (do not repeat them): r1-bulk1d-ring-r16s16 (bulk1d/persistent, cuda_cpp); r2-ldg256-tokblk (ldg256/oneshot, cuda_cpp); r3-os-l2pf-r8w4 (ldg128/oneshot, cuda_cpp); r3-tokblk-s-disp (ldg128/oneshot, triton); r4-cuda-ldg256-token-r2 (ldg256/oneshot, cuda_cpp); r4-cute-ldg256-sw2-disp (ldg256/persistent, cute_dsl); r4-ldg256-wstat-pf2 (ldg256/persistent-wstat, cuda_cpp); r6-gluon-cpasync-os-r16w8-stel (cpasync/oneshot, gluon); r6-ldg256-qkpair-r16 (ldg256/oneshot, cuda_cpp); r7-cap4-qkcat (ldg256/persistent, cuda_cpp); r7-s-r8w4-cta128-disp (ldg256/oneshot, cuda_cpp)."}


## Turn 0 (11794 tokens out, $0.343084)




### probe_b200 (1): inspect probe API

**probe_b200** →
```
probe on NVIDIA B200 (2.1 s), exit code 0
stdout:
cuda_kernel (src, name, opts=())

---
harness_time (fn, inputs=(), outputs=(), warmup=10, rep=50, tries=3)
Median µs of fn(*inputs, *outputs) with the harness's CUPTI methodology (cold L2, shifted pointers).
---
flush_l2 ()
Zero a buffer twice the L2 size, as the harness does before every timed call.
---
def harness_time(fn, inputs=(), outputs=(), warmup=10, rep=50, tries=3):
    """Median µs of fn(*inputs, *outputs) with the harness's CUPTI methodology (cold L2, shifted pointers)."""
    from sol_execbench.core.bench.timing import time_runnable
    for attempt in range(tries):
        try:
            ms = time_runnable(fn, list(inputs), list(outputs), "cuda:0", warmup=warmup, rep=rep,
                               return_mode="median", methodology="cupti")
            return ms * 1e3
        except ValueError as e:                          # CUPTI occasionally drops an iteration's records here
            if "activity sequence" not in str(e) or attempt == tries - 1:
                raise



```


## Turn 1 (239 tokens out, $0.1157022)




### probe_b200 (2): see launcher arg conversion

**probe_b200** →
```
probe on NVIDIA B200 (2.0 s), exit code 0
stdout:
def cuda_kernel(src, name, opts=()):
    return Kernel(compile_cubin(src, opts), name)



```


## Turn 2 (112 tokens out, $0.0202492)




### probe_b200 (3): see Kernel launcher

**probe_b200** →
```
probe on NVIDIA B200 (2.4 s), exit code 0
stdout:
class Kernel:
    def __init__(self, cubin, name):
        torch.zeros(1, device="cuda")                    # make torch's primary context current
        self.module = _ok(driver.cuModuleLoadData(cubin), "cuModuleLoadData")
        self.fn = _ok(driver.cuModuleGetFunction(self.module, name.encode()), "cuModuleGetFunction")
        self.name = name

    def set_smem(self, nbytes):
        """Allow more than 48 KB of dynamic shared memory."""
        attr = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        _ok(driver.cuFuncSetAttribute(self.fn, attr, nbytes), "cuFuncSetAttribute")

    def __call__(self, grid, block, *args, smem=0, stream=None):
        vals, types = [], []
        for a in args:
            if isinstance(a, torch.Tensor):
                vals.append(a.data_ptr()); types.append(ctypes.c_void_p)
            elif isinstance(a, i64):
                vals.append(int(a)); types.append(ctypes.c_int64)
            elif isinstance(a, int):
                vals.append(a); types.append(ctypes.c_int32)
            elif isinstance(a, float):
                vals.append(a); types.append(ctypes.c_float)
            else:
                raise TypeError(f"unsupported launch argument {type(a)}")
        g = tuple(grid) + (1,) * (3 - len(grid))
        b = tuple(block) + (1,) * (3 - len(block))
        s = stream if stream is not None else torch.cuda.current_stream().cuda_stream
        _ok(driver.cuLaunchKernel(self.fn, *g, *b, smem, driver.CUstream(s), (tuple(vals), tuple(types)), 0),
            f"cuLaunchKernel({self.name})")



```


## Turn 3 (2668 tokens out, $0.0755682)




### probe_b200 (4): L2 policy variants for M/L: tail-only evict_last, EF loads, EL weights

**probe_b200** →
```
probe on NVIDIA B200 (5.3 s), exit code 0
stdout:
1024 parent=14.45 ldEF=14.18 wEL=14.58 tail96_plain=14.50 tail64_plain=14.59 tail96_EF=14.56 tail112_EN=14.56
2048 parent=28.50 ldEF=29.14 wEL=28.55 tail96_plain=28.54 tail64_plain=28.50 tail96_EF=28.41 tail112_EN=28.32
4096 parent=57.86 ldEF=58.04 wEL=57.88 tail96_plain=56.82 tail64_plain=57.19 tail96_EF=57.14 tail112_EN=57.24
8192 parent=115.54 ldEF=115.30 wEL=115.76 tail96_plain=115.80 tail64_plain=115.56 tail96_EF=115.83 tail112_EN=115.68


```


## Turn 4 (2212 tokens out, $0.0838546)




### probe_b200 (5): controls: plain stores, pure copy ceiling, QK interleave, CTA size

**probe_b200** →
```
probe on NVIDIA B200 (4.9 s), exit code 0
stdout:
256 parent=5.76 plainst=6.43 copyEL=5.47 interleave=5.73 cta128=5.71 cta512=5.76
1024 parent=14.69 plainst=17.79 copyEL=14.18 interleave=14.95 cta128=14.50 cta512=15.42
2048 parent=28.38 plainst=32.08 copyEL=28.57 interleave=28.82 cta128=28.64 cta512=28.55
4096 parent=57.51 plainst=61.27 copyEL=57.76 interleave=58.28 cta128=57.83 cta512=57.21
8192 parent=115.75 plainst=119.64 copyEL=115.75 interleave=115.93 cta128=115.78 cta512=114.49


```


## Turn 5 (2686 tokens out, $0.090044)




### probe_b200 (6): does dirty flush state matter? read/write-only costs

**probe_b200** →
```
probe on NVIDIA B200 (3.2 s), exit code 0
stdout:
25 MB per side: copyEL/zero=13.3 copyEL/zero+read=11.2 copyEL/zero+sync=13.2 copy/zero=13.3 copy/zero+read=11.2 copy/zero+sync=13.2 read/zero=13.2 read/zero+read=11.3 read/zero+sync=11.2 writeEL/zero=9.2 writeEL/zero+read=9.2 writeEL/zero+sync=9.2 write/zero=9.2 write/zero+read=9.2 write/zero+sync=9.2
50 MB per side: copyEL/zero=23.6 copyEL/zero+read=17.4 copyEL/zero+sync=23.4 copy/zero=21.5 copy/zero+read=17.4 copy/zero+sync=21.4 read/zero=17.4 read/zero+read=15.4 read/zero+sync=17.4 writeEL/zero=13.3 writeEL/zero+read=13.3 writeEL/zero+sync=13.3 write/zero=13.3 write/zero+read=13.3 write/zero+sync=13.3
100 MB per side: copyEL/zero=37.9 copyEL/zero+read=29.7 copyEL/zero+sync=37.9 copy/zero=35.8 copy/zero+read=27.6 copy/zero+sync=35.8 read/zero=27.6 read/zero+read=21.5 read/zero+sync=25.6 writeEL/zero=19.5 writeEL/zero+read=19.5 writeEL/zero+sync=19.4 write/zero=19.5 write/zero+read=19.5 write/zero+sync=19.4
400 MB per side: copyEL/zero=130.1 copyEL/zero+read=117.8 copyEL/zero+sync=130.1 copy/zero=128.0 copy/zero+read=119.8 copy/zero+sync=128.0 read/zero=80.9 read/zero+read=64.6 read/zero+sync=80.9 writeEL/zero=62.5 writeEL/zero+read=60.4 writeEL/zero+sync=62.3 write/zero=60.4 write/zero+read=60.4 write/zero+sync=60.4


```


## Turn 6 (2821 tokens out, $0.1010202)




### probe_b200 (7): which load flavour avoids evicting dirty flush lines (dirty vs clean L2)

**probe_b200** →
```
probe on NVIDIA B200 (3.3 s), exit code 0
stdout:
100 plain=25.7/21.5 cs=25.6/21.5 lu=25.6/21.5 ncEF=23.6/21.5 EFpol=23.6/21.5 EFEFpol=23.6/21.5 EUpol=25.6/21.5 na_EF=23.6/21.5
400 plain=80.9/64.6 cs=80.9/66.5 lu=80.9/66.6 ncEF=72.7/66.6 EFpol=70.8/66.6 EFEFpol=70.7/66.5 EUpol=80.9/66.4 na_EF=70.7/66.6


```


## Turn 7 (5887 tokens out, $0.1597308)




### probe_b200 (8): repeat-measure ldEF and larger CTAs at M/L

**probe_b200** →
```
probe on NVIDIA B200 (8.0 s), exit code 0
stdout:
1024 parent=14.78(14.72-14.79) ldEF=14.59(14.58-14.59) c512=15.36(15.33-15.38) c1024=16.62(16.61-16.63) c512EF=14.88(14.88-14.88)
2048 parent=28.53(28.40-28.64) ldEF=29.11(29.07-29.12) c512=28.29(28.27-28.32) c1024=30.69(30.69-30.72) c512EF=29.25(29.23-29.28)
4096 parent=57.81(57.80-57.97) ldEF=57.98(57.90-57.99) c512=57.65(57.58-57.68) c1024=59.07(59.02-59.11) c512EF=58.00(56.93-58.05)
8192 parent=115.70(115.64-116.01) ldEF=115.49(115.47-115.67) c512=114.62(114.61-114.89) c1024=115.55(114.12-115.61) c512EF=115.22(115.09-115.25)


```


## Turn 8 (6563 tokens out, $0.1994468)




### run_tests (draft 1): r8: dispatch EF x loads at 600<T<=1536, 512-thread 32-row CTAs above

**run_tests** →
```
run_tests on NVIDIA B200 (28.9 s): 16/16 workloads PASSED
latency vs reference r6-s-ldef-nc-disp on the same GPU (µs, this vs reference):
  2,128 5.5 vs 5.6 (-2.0%); 4,1657 92.8 vs 93.5 (-0.8%); 4,1024 58.0 vs 58.0 (+0.1%); 8,773 86.5 vs 87.1 (-0.8%); 8,128 14.2 vs 14.4 (-1.5%); 2,293 8.6 vs 8.6 (-0.3%); 16,256 57.9 vs 58.0 (-0.2%); 1,1024 14.6 vs 14.8 (-1.2%); 4,256 14.1 vs 14.4 (-1.9%); 1,128 4.1 vs 4.1 (+0.7%); 32,128 56.5 vs 58.1 (-2.8%); 4,512 28.5 vs 28.6 (-0.4%); 1,131 4.1 vs 4.1 (-1.7%); 8,256 28.3 vs 28.5 (-0.8%); 1,8192 112.8 vs 115.7 (-2.5%); 1,256 5.5 vs 5.3 (+2.8%)
change by band (geomean): S -0.1%, M -1.2%, L -1.2%
projected portal score: 0.6131 (current best 0.6099). This is a real B200 with unlocked clocks; differences at S/M come out larger than on the portal, so the projection shrinks them (portal change ~ measured change ** k, k = S 0.42, M 0.64, L 1.02, fitted on kernels measured on both).
```


## Turn 9 (6950 tokens out, $0.2156384)

The candidate is r8-ldg256-l2mode-disp. On the rented B200 it is about 1% faster than r6-s-ldef-nc-disp on medium and large inputs, against the 5% the task asked for. It passes all 16 workloads, the harness projects a portal score of 0.613 (best so far 0.610), and that gain is close to noise. My probes suggest the parent is already as fast as a plain copy for these sizes, so I don't think any single change can cut 5% here.

**What the probes found (rented B200, harness timing):**
- **The parent matches a plain copy.** A 256-bit copy with evict_last stores and no maths times the same as the parent at 1024–8192 tokens (14.2–115.8 µs). Medium and large inputs are limited by DRAM bandwidth at about 7.05 TB/s for reads plus writes, not by the kernel. The evict_last effect does reproduce: plain stores are 4–21% slower.
- **The harness's L2 flush leaves dirty lines.** Its zero-fill fills the L2 with about 120 MB of dirty lines, and our kernel pays to write them back. A 400 MB read-only kernel takes 80.9 µs after a zero flush and 64.6 µs after a clean one. Evict_first loads recover part of that (70.7 µs) in a read-only kernel.
- **Why the L2 end state can't fix it.** Whatever dirty output stays in L2 at the end displaced a dirty flush line earlier, so the write traffic inside the timed window is at least the output size. Changing what sits in L2 at the end cannot get below reads plus writes, and the tests agree. Several L2-policy variants all landed within ±1% of the parent:
  - evict_last only on the last 64/96/112 MB of output;
  - evict_first or normal priority on the earlier output;
  - evict_last weights;
  - interleaving Q and K tiles;
  - 128- or 1024-thread CTAs.
- **The two small gains that repeated,** measured three times each:
  - evict_first on x loads at 1024 tokens: −1.3%;
  - 512-thread CTAs with 32 rows each at 2048 tokens and above: −0.3% to −0.9%.

The candidate dispatches between those two by token count. Small inputs run r6's path unchanged. I did not run `compile_b200` on the final files, so registers and spills are not checked. The design-card figures are from the parent (31 registers), and I expect the 512-thread variant to stay under 32.

### Rationale
r8-ldg256-l2mode-disp keeps r6's half-warp-per-row, 256-bit, evict_last-store structure and changes only, by token count, the load cache policy and the CTA size. Given the copy ceiling above, it takes the only two changes that repeated in probes. Full test: medium −1.2%, large −1.2%, small −0.1%.

```json solution-spec
{"languages": ["cuda_cpp"], "entry_point": "binding.cpp::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}
```

```cuda file=kernel.cu
// r8-ldg256-l2mode-disp: per-head RMSNorm of Q and K (fp32), one fused one-shot launch, size dispatch.
// Structure of r5/r6: half-warp per 128-float row, 256-bit ld/st on sm_100a, output stores L2 evict_last.
// Dispatch (shape only):
//   B*S <= 600          : 16 rows / 256-thread CTA, x loads L2 evict_first, w ld.global.nc (r6 S path)
//   600 < B*S <= 1536   : 16 rows / 256-thread CTA, x loads L2 evict_first, w default
//   B*S > 1536          : 32 rows / 512-thread CTA, default loads (fewer, fatter CTAs)
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int D = 128;
constexpr int H = 48;
constexpr uint64_t POL_EVICT_FIRST = 0x12F0000000000000ULL;
constexpr uint64_t POL_EVICT_LAST  = 0x14F0000000000000ULL;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__device__ __forceinline__ void ld8(const float* p, float (&v)[8]) {
    asm volatile("ld.global.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]) : "l"(p));
}
__device__ __forceinline__ void ld8_nc(const float* p, float (&v)[8]) {
    asm volatile("ld.global.nc.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]) : "l"(p));
}
__device__ __forceinline__ void ld8_h(const float* p, float (&v)[8], uint64_t pol) {
    asm volatile("ld.global.L2::cache_hint.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]) : "l"(p), "l"(pol));
}
__device__ __forceinline__ void st8_h(float* p, const float (&v)[8], uint64_t pol) {
    asm volatile("st.global.L2::cache_hint.v8.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
                 :: "l"(p), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]),
                    "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]), "l"(pol) : "memory");
}
#else
__device__ __forceinline__ void ld4(const float* p, float* v) {
    asm volatile("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "l"(p));
}
__device__ __forceinline__ void ld4_nc(const float* p, float* v) {
    asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "l"(p));
}
__device__ __forceinline__ void ld4_h(const float* p, float* v, uint64_t pol) {
    asm volatile("ld.global.L2::cache_hint.v4.f32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "l"(p), "l"(pol));
}
__device__ __forceinline__ void st4_h(float* p, const float* v, uint64_t pol) {
    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1,%2,%3,%4}, %5;"
                 :: "l"(p), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "l"(pol) : "memory");
}
__device__ __forceinline__ void ld8(const float* p, float (&v)[8]) { ld4(p, v); ld4(p + 4, v + 4); }
__device__ __forceinline__ void ld8_nc(const float* p, float (&v)[8]) { ld4_nc(p, v); ld4_nc(p + 4, v + 4); }
__device__ __forceinline__ void ld8_h(const float* p, float (&v)[8], uint64_t pol) {
    ld4_h(p, v, pol); ld4_h(p + 4, v + 4, pol);
}
__device__ __forceinline__ void st8_h(float* p, const float (&v)[8], uint64_t pol) {
    st4_h(p, v, pol); st4_h(p + 4, v + 4, pol);
}
#endif

// XMODE: 0 default x loads, 1 evict_first x loads. WNC: weights via ld.global.nc.
template <int THREADS, int XMODE, bool WNC>
__global__ void __launch_bounds__(THREADS, 2048 / THREADS)
qk_rms_kernel(const float* __restrict__ q, const float* __restrict__ k,
              const float* __restrict__ wq, const float* __restrict__ wk,
              float* __restrict__ qo, float* __restrict__ ko, int n_tiles, float eps) {
    constexpr int ROWS = THREADS / 16;
    const int bid = blockIdx.x;
    const bool is_k = bid >= n_tiles;
    const int tile = is_k ? bid - n_tiles : bid;
    const float* __restrict__ x = is_k ? k : q;
    const float* __restrict__ w = is_k ? wk : wq;
    float* __restrict__ y = is_k ? ko : qo;

    const int r = threadIdx.x >> 4;
    const int c = (threadIdx.x & 15) * 8;
    const size_t row = (size_t)tile * ROWS + r;
    const int head = (int)(row % H);

    float xv[8], wv[8], yv[8];
    if (XMODE == 1) ld8_h(x + row * D + c, xv, POL_EVICT_FIRST);
    else            ld8(x + row * D + c, xv);
    if (WNC) ld8_nc(w + (size_t)head * D + c, wv);
    else     ld8(w + (size_t)head * D + c, wv);

    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) s = fmaf(xv[i], xv[i], s);
    s += __shfl_xor_sync(0xffffffffu, s, 8);
    s += __shfl_xor_sync(0xffffffffu, s, 4);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    const float inv = rsqrtf(s * (1.0f / (float)D) + eps);   // * (1/128) is exact
#pragma unroll
    for (int i = 0; i < 8; ++i) yv[i] = (xv[i] * inv) * wv[i];
    st8_h(y + row * D + c, yv, POL_EVICT_LAST);
}
}  // namespace

extern "C" int qk_rms_launch(const float* q, const float* k, const float* wq, const float* wk,
                             float* qo, float* ko, long long n_rows, float eps, cudaStream_t stream) {
    if (n_rows <= 0) return 0;
    const long long tokens = n_rows / H;
    if (tokens <= 600) {
        const long long nt = n_rows / 16;
        qk_rms_kernel<256, 1, true><<<(unsigned)(2 * nt), 256, 0, stream>>>(q, k, wq, wk, qo, ko, (int)nt, eps);
    } else if (tokens <= 1536 || (n_rows % 32) != 0) {
        const long long nt = n_rows / 16;
        if (tokens <= 1536)
            qk_rms_kernel<256, 1, false><<<(unsigned)(2 * nt), 256, 0, stream>>>(q, k, wq, wk, qo, ko, (int)nt, eps);
        else
            qk_rms_kernel<256, 0, false><<<(unsigned)(2 * nt), 256, 0, stream>>>(q, k, wq, wk, qo, ko, (int)nt, eps);
    } else {
        const long long nt = n_rows / 32;
        qk_rms_kernel<512, 0, false><<<(unsigned)(2 * nt), 512, 0, stream>>>(q, k, wq, wk, qo, ko, (int)nt, eps);
    }
    return (int)cudaGetLastError();
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>

extern "C" int qk_rms_launch(const float* q, const float* k, const float* wq, const float* wk,
                             float* qo, float* ko, long long n_rows, float eps, cudaStream_t stream);

static void check_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous(), name,
                " must be a contiguous fp32 CUDA tensor");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(t.data_ptr()) & 31) == 0, name, " must be 32-byte aligned");
}

void run(torch::Tensor query, torch::Tensor key, torch::Tensor weight_q, torch::Tensor weight_k,
         double eps, torch::Tensor query_norm, torch::Tensor key_norm) {
    check_tensor(query, "query"); check_tensor(key, "key");
    check_tensor(weight_q, "weight_q"); check_tensor(weight_k, "weight_k");
    check_tensor(query_norm, "query_norm"); check_tensor(key_norm, "key_norm");
    TORCH_CHECK(weight_q.numel() == 48 * 128 && weight_k.numel() == 48 * 128, "weights must be [48,128]");
    TORCH_CHECK(query.numel() % (48 * 128) == 0 && key.numel() == query.numel() &&
                query_norm.numel() == query.numel() && key_norm.numel() == query.numel(), "shape mismatch");
    const int64_t n_rows = query.numel() / 128;
    TORCH_CHECK(n_rows / 16 * 2 <= (int64_t)0x7fffffff, "grid too large");
    const c10::cuda::CUDAGuard guard(query.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int err = qk_rms_launch(query.data_ptr<float>(), key.data_ptr<float>(), weight_q.data_ptr<float>(),
                            weight_k.data_ptr<float>(), query_norm.data_ptr<float>(), key_norm.data_ptr<float>(),
                            (long long)n_rows, (float)eps, stream);
    TORCH_CHECK(err == 0, "qk_rms launch failed: ", cudaGetErrorString((cudaError_t)err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Fused per-head RMSNorm of Q and K (fp32), size-dispatched L2 policy / CTA size");
}
```

```yaml design-card
id: r8-ldg256-l2mode-disp
parents:
  - r6-s-ldef-nc-disp
operation: structural_mutation
language: cuda_cpp
niche:
  mem: ldg256
  st: direct
  grid: oneshot
  launch: fused
  tile: rows32
  red: halfwarp
  cache: stream
  spec: dispatch:size
hypothesis: >-
  Probes on the rented B200 (harness CUPTI timing, cold L2) show the parent is at the ceiling for medium and
  large inputs. A 256-bit copy with evict_last stores and no maths times the same as the parent
  (14.2/28.6/57.8/115.8 us at 1024/2048/4096/8192 tokens). The harness zero-flush leaves about 120 MB of dirty
  L2 lines; a 400 MB read-only kernel takes 80.9 us after a zero flush against 64.6 us after a clean flush.
  Any dirty output left in L2 at kernel end displaced a dirty flush line earlier, so in-window DRAM traffic
  cannot drop below reads + writes. That explains why these L2-policy variants were all within +-1%:
  evict_last only on the last 64/96/112 MB of output, evict_first or normal stores for the early output,
  evict_last weights, Q/K tile interleave, and 128- or 1024-thread CTAs. Two small effects repeated (3x each)
  and are dispatched here: evict_first x loads at 1024 tokens (-1.3%), and 512-thread CTAs with 32 rows for
  more than 1536 tokens (-0.3 to -0.9%). Small inputs keep r6's path. Full run_tests vs the parent on the
  rented B200: S -0.1%, M -1.2%, L -1.2%, 16/16 pass, projected portal score 0.6131 (parent 0.6099).
  The gain is B200-specific and close to noise.
expected_effect:
  S:
    pct: 0
    confidence: high
  M:
    pct: -0.8
    confidence: low
  L:
    pct: -1.0
    confidence: low
resources_sm100a:
  regs_per_thread: 31
  smem_per_cta_bytes: 0
  threads_per_cta: 512
  ctas_per_sm: 4
  bytes_in_flight_per_sm: 65536
  launches_per_call: 1
knobs:
  S_MAX_TOKENS: 600
  EF_MAX_TOKENS: 1536
  L_THREADS: 512
  L_ROWS: 32
  store_policy: "L2 evict_last constant 0x14F0000000000000 (all sizes)"
dispatch:
  - max_tokens: 600
    kernel: qk_rms_kernel<256,1,true>
    meta:
      ROWS: 16
      loads: "x evict_first, w nc"
  - max_tokens: 1536
    kernel: qk_rms_kernel<256,1,false>
    meta:
      ROWS: 16
      loads: "x evict_first, w default"
  - max_tokens: 1000000000
    kernel: qk_rms_kernel<512,0,false>
    meta:
      ROWS: 32
      loads: "default"
      note: "falls back to the 256-thread kernel if n_rows % 32 != 0"
runs_on:
  H200:
    runs: true
    representative: false
    note: "128-bit fallback, 50 MB L2"
  A100:
    runs: true
    representative: false
  L40S:
    runs: true
    representative: false
risks:
  - "Gains of about 1% are at the edge of portal noise (about 5%); the portal may show no change."
  - "Medium and large sizes appear bound by reads+writes at about 7.05 TB/s; further structural changes to this kernel are unlikely to help."
  - "compile_b200 was not run on the final files; regs_per_thread is the parent's value, not a measured one."
measure_first:
  - H200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```


---
session total: 10 turns, 1 GPU calls, $1.40
