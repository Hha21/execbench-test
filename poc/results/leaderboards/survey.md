# SOL-ExecBench public leaderboard survey (B200, evaluation stack v1.1)

Data: public endpoints `/api/kernels` and `/api/leaderboard/kernel/<id>/B200?evaluation_stack_version=v1.1`, fetched 2026-10-09 via WebFetch (a small summarising model parsed the JSON; three boards, ids 81, 46 and 171, needed a re-query because the first parse was malformed, and id 17 needed a re-query for usernames). Numbers are as reported by the fetch tool. Full data: `leaderboards_b200_v1.1.csv`.

**Kernels fetched: 235 of 235** (leaderboards for all 235 ids returned data).

## Kernels with n_ranked < 5

**None.** Every board has at least 15 ranked entries (range 15 to 33). There is no free entry into the top 5; the baseline (0.5) only qualifies where s5 is at or below 0.5 (see id 30).

Fewest ranked entries (for reference):

| id | name | n_ranked | s1 | s5 |
|---|---|---|---|---|
| 135 | 041_kv_shared_attention_with_dual_rope | 15 | 0.933526 | 0.901369 |
| 152 | 058_mamba2_selective_scan | 15 | 0.818886 | 0.775977 |
| 157 | 063_encoder_layer_dual_residual_norm_chain | 15 | 0.670133 | 0.626916 |
| 159 | 065_sparse_expert_dispatch_and_combine | 15 | 0.717874 | 0.698222 |
| 166 | 072_region_aware_self_attention_with_edit_bias_backward | 15 | 0.83888 | 0.807089 |
| 101 | 007_multimodal_rotary_embedding_attention | 16 | 0.6675 | 0.632461 |
| 102 | 008_moe_sparse_routing_and_dispatch | 16 | 0.967905 | 0.901832 |
| 103 | 009_decoder_layer_with_residual_connections | 16 | 0.729853 | 0.70659 |
| 106 | 012_moe_expert_batched_execution_with_capacity_factor | 16 | 0.954793 | 0.939108 |
| 107 | 013_expert_weighted_aggregation_with_shared_expert | 16 | 0.882648 | 0.822381 |
| 109 | 015_audio_sinusoidal_position_embedding_with_conv_projection | 16 | 0.839038 | 0.791939 |
| 111 | 017_fused_vision_cu_seqlens_attention_with_2d_rope_backward | 16 | 0.82491 | 0.799618 |

## Top-25 opportunities

Ordering mixes bar (s5), headroom (SOL speedup) and fit with memory-bound/fusion work. SOL speedup = SOL Bound avg_speedup over the PyTorch reference. Many of these kernels have reference latencies of 0.01-0.1 ms, so scores there are launch-overhead and noise sensitive (the operator already saw 2-3% run-to-run variation).

| id | name | tags | n_ranked | s1 | s5 | SOL speedup | why promising | rough difficulty |
|---|---|---|---|---|---|---|---|---|
| 30 | 030_attention_output_projection_with_residual | other;L1 | 24 | 0.54638 | 0.490871 | 1.8037 | s5 is 0.491, below the 0.5 baseline: matching the baseline already ranks 5th; >0.531 gives top 3. Small GEMM + residual add (ref 0.047 ms). | Easy (fused GEMM-epilogue residual); headroom only 1.8x |
| 84 | 084_silu_activation_backward | mlp;L1 | 25 | 0.560423 | 0.537483 | 5.783 | Pure elementwise backward (SiLU grad); s5 0.537 and SOL is 5.8x the reference. Ref only 0.03 ms so launch overhead dominates. | Easy |
| 10 | 010_attention_value_projection_with_transpose | other;L1 | 26 | 0.534899 | 0.523347 | 2.4848 | s5 0.523, barely above baseline; small projection + transpose, SOL 2.5x. | Easy-Medium (small GEMM, layout fusion) |
| 129 | 035_convnextv2_block_with_grn | other;L2 | 16 | 0.690412 | 0.619395 | 52.4109 | GRN block (reduction + elementwise); s5 0.619 yet SOL is 52x the reference, so fusion headroom is huge; only 16 ranked. | Medium (fused reduce + scale kernels) |
| 49 | 049_attention_qk_matmul_with_gqa_repeat_and_scaling | other;L1 | 26 | 0.61776 | 0.596654 | 5.2314 | QK^T with GQA repeat and scaling; s5 0.597, SOL 5.2x; ref 0.17 ms, memory-bound if repeat is fused away. | Medium |
| 31 | 031_repeat_kv_attention_matmul | attention;L1 | 23 | 0.610167 | 0.592013 | 3.127 | repeat_kv + attention matmul; s5 0.592, SOL 3.1x; avoiding materialised repeat_kv is the win. | Medium |
| 157 | 063_encoder_layer_dual_residual_norm_chain | normalization;decoder;L2 | 15 | 0.670133 | 0.626916 | 6.2299 | Dual residual + norm chain; only 15 ranked, s5 0.627, SOL 6.2x. Norm/residual fusion fits the team strength. | Easy-Medium |
| 38 | 038_flux_multi_head_rmsnorm_qk | normalization;diffusion;L1 | 28 | 0.627525 | 0.609159 | 3.4541 | multi-head RMSNorm for QK; s5 0.609, s1-s5 spread only 0.018; already iterated by the operator (0.6125 would sit around rank 4). | Medium (bar is tight, gains are 1-3%) |
| 162 | 068_gelu_approximate_feedforward_backward | mlp;L2 | 16 | 0.665242 | 0.611143 | 3.6511 | GELU-approx FFN backward; s5 0.611, SOL 3.7x; elementwise grads around two GEMMs. | Medium |
| 173 | 079_git_vision_encoder_layer | vision;decoder;L2 | 19 | 0.61494 | 0.587402 | 6.6894 | Vision encoder layer (norm/residual/GELU around GEMMs); s5 0.587, SOL 6.7x. | Medium-Hard (full layer) |
| 164 | 070_basic_transformer_block | other;L2 | 19 | 0.595235 | 0.572991 | 4.7706 | Basic transformer block; s5 0.573, SOL 4.8x; many fusable norm/elementwise ops. | Medium-Hard (full block) |
| 36 | 036_flux_output_norm_projection_chain | normalization;diffusion;L1 | 21 | 0.723995 | 0.652592 | 9.4193 | Output norm + projection chain; s5 0.653, SOL 9.4x; norm modulation fusion. | Medium |
| 161 | 067_patch_embed_to_joint_attention_input | vision;L2 | 17 | 0.702171 | 0.633185 | 9.9117 | Patch embed to joint attention input; s5 0.633, SOL 9.9x; layout/concat/norm fusion. | Medium |
| 127 | 033_multi_scale_feature_pyramid | vision;L2 | 16 | 0.755214 | 0.663448 | 45.2235 | Multi-scale feature pyramid; s5 0.663, SOL 45x (ref 0.99 ms, SOL 0.019 ms): large fusion headroom, 16 ranked. | Medium (conv/resample, check shapes) |
| 148 | 054_vision_encoder_layer_with_gated_residuals | vision;decoder;L2 | 18 | 0.667227 | 0.637199 | 4.6262 | Vision encoder layer with gated residuals; s5 0.637, SOL 4.6x. | Medium-Hard |
| 53 | 053_gaussian_topk_sparse_activation | moe;L1 | 20 | 0.67965 | 0.654555 | 1.8007 | Gaussian top-k sparse activation; s5 0.655, SOL only 1.8x but top-k/elementwise is in the team's comfort zone. | Medium |
| 25 | 025_video_latent_gelu_activation | mlp;video;L1 | 21 | 0.677856 | 0.639126 | 1.6392 | Video latent GELU; s5 0.639; SOL only 1.6x (ref 0.02 ms), noise-limited. | Easy but low headroom |
| 128 | 034_vision_language_cross_attention_fusion | vision;L2 | 17 | 0.653375 | 0.627281 | 3.1522 | Vision-language cross-attention fusion; s5 0.627, tight spread (s1-s5 0.026); SOL 3.2x. | Hard (attention) |
| 158 | 064_multi_head_qkv_projection_with_rope_backward | rope;L2 | 17 | 0.691548 | 0.658294 | 5.0457 | QKV projection + RoPE backward; s5 0.658, SOL 5.0x; RoPE grad is memory-bound. | Medium |
| 124 | 030_flux_concatenated_sequence_processing_with_split | diffusion;L2 | 17 | 0.709499 | 0.671234 | 4.133 | Concatenated sequence processing with split; s5 0.671, SOL 4.1x; concat/split copies are bandwidth work. | Easy-Medium |
| 85 | 085_geglu_activation | mlp;L1 | 23 | 0.690519 | 0.680444 | 4.9113 | GeGLU; s5 0.680, SOL 4.9x, pure elementwise (tiny ref 0.068 ms). | Easy, but crowded bar (spread s1-s5 0.010) |
| 121 | 027_grouped_query_attention_with_yarn_rope_and_qk_norm | normalization;rope;L2 | 17 | 0.642898 | 0.55836 | 2.5448 | GQA + YaRN RoPE + QK-norm; s5 0.558, s1 0.643; SOL 2.5x. | Hard (attention) |
| 92 | 092_gqa_attention_with_qk_norm | normalization;L1 | 21 | 0.687177 | 0.578942 | 2.2259 | GQA attention with QK-norm; s5 0.579, SOL 2.2x. | Hard (attention) |
| 15 | 015_grouped_query_attention_with_rope_and_qk_norm | normalization;rope;L1 | 24 | 0.702299 | 0.624993 | 3.0035 | GQA with RoPE and QK-norm; s5 0.625, SOL 3.0x. | Hard (attention) |
| 96 | 002_decoder_layer_full_block | decoder;L2 | 18 | 0.628328 | 0.522872 | 2.5508 | Decoder layer full block; s5 0.523 (s10 0.485), 18 ranked, SOL 2.5x, ref 21 ms. Low bar but needs a whole fast layer. | Hard (GEMM + attention) |
| 119 | 025_moe_expert_parallel_execution_backward | moe;L2 | 21 | 0.535106 | 0.528871 | 7.6373 | MoE expert backward; s5 0.529, s1 0.535 (nobody has beaten baseline by much), SOL 7.6x, ref 3.7 s. | Hard (grouped GEMM backward) |

## Lowest s5 across all boards

| id | name | n_ranked | s1 | s5 | s10 | SOL speedup | rank-1 date |
|---|---|---|---|---|---|---|---|
| 30 | 030_attention_output_projection_with_residual | 24 | 0.54638 | 0.490871 | 0.460148 | 1.8037 | 2026-08-20 |
| 96 | 002_decoder_layer_full_block | 18 | 0.628328 | 0.522872 | 0.484923 | 2.5508 | 2026-07-14 |
| 10 | 010_attention_value_projection_with_transpose | 26 | 0.534899 | 0.523347 | 0.498515 | 2.4848 | 2026-08-29 |
| 218 | 009_gemm_n5120_k2048 | 22 | 0.53563 | 0.52514 | 0.506841 | 2.3157 | 2026-07-20 |
| 119 | 025_moe_expert_parallel_execution_backward | 21 | 0.535106 | 0.528871 | 0.526835 | 7.6373 | 2026-07-22 |
| 3 | 003_lm_head_projection_with_logit_slicing | 28 | 0.543467 | 0.532958 | 0.475001 | 1.238 | 2026-09-16 |
| 219 | 010_gemm_n6144_k4096 | 24 | 0.586722 | 0.5362 | 0.515032 | 2.38 | 2026-10-06 |
| 84 | 084_silu_activation_backward | 25 | 0.560423 | 0.537483 | 0.524736 | 5.783 | 2026-09-26 |
| 113 | 019_decoder_layer_fused_attention_mlp | 18 | 0.67574 | 0.54174 | 0.519442 | 2.5249 | 2026-07-28 |
| 216 | 007_gemm_n4096_k4096 | 26 | 0.586421 | 0.548795 | 0.519369 | 2.7473 | 2026-10-07 |
| 121 | 027_grouped_query_attention_with_yarn_rope_and_qk_norm | 17 | 0.642898 | 0.55836 | 0.528162 | 2.5448 | 2026-10-07 |
| 164 | 070_basic_transformer_block | 19 | 0.595235 | 0.572991 | 0.54127 | 4.7706 | 2026-09-26 |
| 92 | 092_gqa_attention_with_qk_norm | 21 | 0.687177 | 0.578942 | 0.549157 | 2.2259 | 2026-09-29 |
| 215 | 006_gemm_n2048_k4096 | 23 | 0.642039 | 0.580085 | 0.544904 | 3.1961 | 2026-10-06 |
| 173 | 079_git_vision_encoder_layer | 19 | 0.61494 | 0.587402 | 0.534546 | 6.6894 | 2026-10-06 |
| 217 | 008_gemm_n4096_k14336 | 23 | 0.690203 | 0.591225 | 0.552042 | 2.3783 | 2026-10-06 |
| 31 | 031_repeat_kv_attention_matmul | 23 | 0.610167 | 0.592013 | 0.53343 | 3.127 | 2026-10-07 |
| 49 | 049_attention_qk_matmul_with_gqa_repeat_and_scaling | 26 | 0.61776 | 0.596654 | 0.550425 | 5.2314 | 2026-10-06 |
| 38 | 038_flux_multi_head_rmsnorm_qk | 28 | 0.627525 | 0.609159 | 0.60286 | 3.4541 | 2026-10-02 |
| 162 | 068_gelu_approximate_feedforward_backward | 16 | 0.665242 | 0.611143 | 0.546009 | 3.6511 | 2026-08-19 |

## Staleness

Oldest rank-1 dates (no board is empty or abandoned; all have 15+ entries):

- id 28 028_hybrid_attention_mask_preparation: rank 1 set 2026-07-10, s1 0.944046, s5 0.920589
- id 51 051_attention_qkv_with_qk_norm_single_kernel_backward: rank 1 set 2026-07-11, s1 0.996768, s5 0.861542
- id 96 002_decoder_layer_full_block: rank 1 set 2026-07-14, s1 0.628328, s5 0.522872
- id 11 011_rotary_position_embedding: rank 1 set 2026-07-14, s1 0.792246, s5 0.766252
- id 67 067_flash_attention_gqa_ultralong: rank 1 set 2026-07-16, s1 0.69289, s5 0.633639
- id 218 009_gemm_n5120_k2048: rank 1 set 2026-07-20, s1 0.53563, s5 0.52514
- id 119 025_moe_expert_parallel_execution_backward: rank 1 set 2026-07-22, s1 0.535106, s5 0.528871
- id 172 078_fused_final_layer_upsample_with_adaptive_norm: rank 1 set 2026-07-26, s1 0.821643, s5 0.748218

## Users in the top 5 (count of kernels, of 235)

| username | boards in top 5 |
|---|---|
| Amir M. Mir | SF Tensor | 213 |
| Databricks | 198 |
| Geometric | 177 |
| RIAC as well | 169 |
| Infinigence AI | 109 |
| L2@Naive-N0.5-Flash | 51 |
| Zhongzhu Zhou | 31 |
| ac4k | 28 |
| Hyra | 22 |
| Zoom ZAS | 20 |
| zzyyss | 19 |
| L1@Naive-N0.5-Flash | 19 |
| Recursive | 18 |
| Proud Wren @ NVIDIA | 15 |
| Pandeng Yao @ Baidu | 9 |
| doubleAI | 8 |
| Rapid Tiger | 5 |
| Electric Cheetah | 5 |
| AILabs | 5 |
| Gleaming Coyote | 4 |
| Leshan200003 | 4 |
| Proud Wren @ Nvidia | 4 |
| Northern Dark Pretrainer | 4 |
| Hollow Jellyfish | 3 |
| Golden Cobra | 3 |
| Qwen | 3 |
| Proud Wren @NVIDIA | 2 |
| Sam Mayle | 2 |
| Active Model | 2 |
| Iron Penguin | 2 |
| jonathanc.net | 2 |
| Stellar Flamingo | 2 |
| AM I RSI? | 1 |
| Wicked Wren | 1 |
| AccountForCollege | 1 |
| Q H | 1 |
| Proud Wren | 1 |
| asgard-alpha | 1 |
| Ashraf Michail | 1 |
| Deft Koala | 1 |
| Jensen | 1 |
| Linyue Pan @ THU | 1 |
| Weicheng-Gu1 | 1 |
| Calm Otter | 1 |
| Runic Kingfisher | 1 |
| FRANKENBRAIN | 1 |
| Qin Liu @ UCD | 1 |
| Knight-Solaire-590 | 1 |
| Vibe Kernel | 1 |

Note: 'Proud Wren @ NVIDIA', '@NVIDIA', '@ Nvidia' appear as separate strings (an NVIDIA account, total 21 if merged).

## Caveats
- Fields were extracted by a summarising model; spot re-queries matched, but treat the 4th decimal as approximate.
- n_ranked is the count of rows with a numeric rank on the fetched leaderboard.
- Sol-latency values for tiny kernels (down to 1e-5 ms) are below any realistic launch time, so the SOL speedup overstates what is reachable there.
