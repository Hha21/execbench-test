// sglang_flux2_qknorm_excerpt.cuh
//
// Source:  https://github.com/sgl-project/sglang/blob/0b635266d4a09f8db2d12bdcb793b085199faa8f/python/sglang/kernels/kda_kernels/csrc/diffusion/flux2_qkv_epilogue.cuh
//          ("Kernel Design Agent" kernel, SGLang PR #37162; registry:
//          https://github.com/sgl-project/sglang/blob/0b635266d4a09f8db2d12bdcb793b085199faa8f/python/sglang/kernels/ops/diffusion/README.md)
// Commit:  0b635266d4a0 (2026-10-07)
// Licence: Apache-2.0 (SGLang repository licence). Excerpt, unmodified except that lines are elided ("...")
//          and comments starting "// [solx]" are ours.
//
// Why it matters for #38:
// - This is the production FLUX.2 QK RMSNorm (+RoPE, +QKV packing) in SGLang, i.e. the op #38 was extracted
//   from, as written by an agentic kernel generator. It is bf16 and fused with RoPE, so it is a structural
//   reference, not a speed target: one WARP per (token, head) row of 128, 4 elements per lane (8 B for bf16),
//   256-thread CTAs, persistent grid-stride loop over all (Q|K|V, token, head) jobs.
// - Grid = min(occupancy x SM count, jobs / 8): the common "fill the GPU once, loop" pattern. Our B200 data
//   say this loses to one-shot grids for #38 (v039 persistent 0.531 vs one-shot 0.577-0.588), unless the loop
//   keeps several rows' loads in flight per thread.
// - No cache hints, no PDL, plain vector loads/stores; weights re-read per row (L1/L2 hits).

constexpr int kHeadDim = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / device::kWarpThreads;
constexpr int kElemsPerThread = kHeadDim / device::kWarpThreads;
...
__global__ void flux2_qkv_epilogue_kernel(const Params __grid_constant__ params) {
  using namespace device;
  using Packed = packed_t<bf16_t>;
  using Storage = AlignedVector<Packed, kVecSize>;

  const uint32_t lane = threadIdx.x % kWarpThreads;
  const uint32_t warp = threadIdx.x / kWarpThreads;
  const uint32_t start = blockIdx.x * kWarps + warp;
  const uint32_t workers = gridDim.x * kWarps;
  const uint32_t total_tokens = params.txt_tokens + params.img_tokens;
  const uint32_t token_head_works = total_tokens * params.num_heads;
  const uint32_t total_works = 3 * token_head_works;

  for (uint32_t work = start; work < total_works; work += workers) {
    const uint32_t kind = work / token_head_works;  // 0: Q, 1: K, 2: V.
    ...
    void* output =
        pointer::offset(output_base, joint_token * params.output_token_stride_bytes, head * params.head_stride_bytes);

    auto input_vec = load_as<Storage>(input, lane);
    if (kind == 2) {
    ...
    if (kind == 0) {
      weight_base = is_text ? params.txt_q_weight : params.img_q_weight;
    } else {
      weight_base = is_text ? params.txt_k_weight : params.img_k_weight;
    }
    const auto weight_vec = load_as<Storage>(weight_base, lane);

    float elems[kElemsPerThread];
    float sum_of_squares = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      const auto [x0, x1] = cast<fp32x2_t>(input_vec[j]);
      elems[2 * j] = x0;
      elems[2 * j + 1] = x1;
      sum_of_squares += x0 * x0 + x1 * x1;
    }
    sum_of_squares = warp::reduce_sum(sum_of_squares);
    const float eps = is_text ? params.txt_eps : params.img_eps;
    const float norm_factor = math::rsqrt(sum_of_squares / static_cast<float>(kHeadDim) + eps);

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
    ...
    // [solx] RoPE and pack/store elided
  }
}
...

    const uint32_t sm_count = runtime::get_sm_count(device.unwrap().device_id);
    static const uint32_t blocks_per_sm = runtime::get_blocks_per_sm(flux2_qkv_epilogue_kernel, kThreads);
    const uint32_t needed_blocks = div_ceil(total_works, uint32_t(kWarps));
    const uint32_t blocks = std::min(blocks_per_sm * sm_count, needed_blocks);
    LaunchKernel(blocks, kThreads, device.unwrap())(flux2_qkv_epilogue_kernel, params);
