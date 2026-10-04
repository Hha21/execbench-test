// Compile-only probe (nvcc -arch=sm_100a -cubin): CUDA C++ recipes for a memory-bound row-wise kernel.
//  k_ldg256 : one warp per row pair, 256-bit global loads/stores (ld.global.v8.f32, sm_100+ only)
//  k_bulk   : persistent CTA, STAGES-deep ring of 1D bulk copies (cp.async.bulk, no tensor map),
//             mbarrier completion, L2 evict_first policy, bulk store from shared memory.
#include <cstdint>
#include <cuda_runtime.h>

#define D 128
#define H 48

__device__ __forceinline__ void ld256(const float* p, float (&v)[8]) {
  asm volatile("ld.global.L1::no_allocate.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7])
               : "l"(p));
}
__device__ __forceinline__ void st256(float* p, const float (&v)[8]) {
  asm volatile("st.global.v8.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
               :: "l"(p), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7])
               : "memory");
}

// 16 lanes x 8 floats = one 128-float row; a warp handles 2 rows per iteration.
__global__ void __launch_bounds__(256) k_ldg256(const float* __restrict__ q, const float* __restrict__ wq,
                                                float* __restrict__ qo, int n_rows, float eps) {
  int lane = threadIdx.x & 31, half = lane >> 4, l16 = lane & 15;
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int nwarps = (gridDim.x * blockDim.x) >> 5;
  for (int r = warp * 2 + half; r < n_rows; r += nwarps * 2) {
    float v[8], w[8];
    ld256(q + (size_t)r * D + l16 * 8, v);
    const float4* wp = reinterpret_cast<const float4*>(wq + (r % H) * D + l16 * 8);
    float4 w0 = __ldg(wp), w1 = __ldg(wp + 1);
    w[0] = w0.x; w[1] = w0.y; w[2] = w0.z; w[3] = w0.w; w[4] = w1.x; w[5] = w1.y; w[6] = w1.z; w[7] = w1.w;
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) ss = fmaf(v[i], v[i], ss);
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);  // stays within each 16-lane half
    float inv = rsqrtf(ss * (1.0f / D) + eps);
#pragma unroll
    for (int i = 0; i < 8; i++) v[i] = (v[i] * inv) * w[i];
    st256(qo + (size_t)r * D + l16 * 8, v);
  }
}

// ---------------------------------------------------------------- 1D bulk-copy ring
template <int ROWS, int STAGES>
__global__ void __launch_bounds__(ROWS * 32) k_bulk(const float* __restrict__ x, const float* __restrict__ w,
                                                    float* __restrict__ y, int n_rows, float eps) {
  constexpr int TILE_BYTES = ROWS * D * 4;
  __shared__ alignas(128) float xs[STAGES][ROWS * D];
  __shared__ alignas(128) float ys[2][ROWS * D];
  __shared__ alignas(8) uint64_t bar[STAGES];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n_tiles = n_rows / ROWS;
  const int my_n = (n_tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;

  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; s++)
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[s])));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  auto issue = [&](int i) {  // load tile number i of this CTA into stage i % STAGES
    int s = i % STAGES;
    int t = blockIdx.x + i * gridDim.x;
    uint32_t b = (uint32_t)__cvta_generic_to_shared(&bar[s]);
    uint32_t dst = (uint32_t)__cvta_generic_to_shared(&xs[s][0]);
    const float* src = x + (size_t)t * ROWS * D;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(TILE_BYTES) : "memory");
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                 :: "r"(dst), "l"(src), "r"(TILE_BYTES), "r"(b), "l"(policy) : "memory");
  };
  if (threadIdx.x == 0)
    for (int i = 0; i < STAGES && i < my_n; i++) issue(i);

  for (int i = 0; i < my_n; i++) {
    int s = i % STAGES, t = blockIdx.x + i * gridDim.x;
    uint32_t b = (uint32_t)__cvta_generic_to_shared(&bar[s]);
    uint32_t phase = (i / STAGES) & 1, done = 0;
    while (!done)
      asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                   : "=r"(done) : "r"(b), "r"(phase) : "memory");
    // one warp per row: lane holds 4 consecutive floats
    int r = warp, grow = t * ROWS + r;
    float4 v = reinterpret_cast<const float4*>(&xs[s][r * D])[lane];
    float4 wv = __ldg(reinterpret_cast<const float4*>(w + (grow % H) * D) + lane);
    float ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
    float inv = rsqrtf(ss * (1.0f / D) + eps);
    int bsel = i & 1;
    if (threadIdx.x == 0) asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");  // ys[bsel] free again
    __syncthreads();  // also: every warp has finished reading xs[s]
    if (threadIdx.x == 0 && i + STAGES < my_n) issue(i + STAGES);
    float4 o4 = make_float4(v.x * inv * wv.x, v.y * inv * wv.y, v.z * inv * wv.z, v.w * inv * wv.w);
    reinterpret_cast<float4*>(&ys[bsel][r * D])[lane] = o4;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // make generic-proxy smem writes visible to TMA
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t src = (uint32_t)__cvta_generic_to_shared(&ys[bsel][0]);
      asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;"
                   :: "l"(y + (size_t)t * ROWS * D), "r"(src), "r"(TILE_BYTES), "l"(policy) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }
  }
  if (threadIdx.x == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

template __global__ void k_bulk<8, 4>(const float*, const float*, float*, int, float);
template __global__ void k_bulk<16, 4>(const float*, const float*, float*, int, float);
