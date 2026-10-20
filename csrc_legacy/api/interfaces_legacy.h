#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <torch/csrc/stable/tensor.h>

// Entry points of the frozen pre-2026.09.30 kernels, defined in csrc_legacy/api/*.cpp.
// Like the current side, this one is built against PyTorch's stable C++ ABI, so the entry points
// speak `torch::stable::Tensor` and the op-facing scalars are the int64/double a torch schema
// has. No translation unit includes this header today -- each .cpp declares what it calls -- but
// a declaration that disagreed with its definition would be a silent link error, so the types
// here are the definitions' own.

std::tuple<torch::stable::Tensor, torch::stable::Tensor, std::optional<torch::stable::Tensor>, std::optional<torch::stable::Tensor>>
sparse_attn_decode_interface(
    const torch::stable::Tensor &q,
    const torch::stable::Tensor &kv,
    const torch::stable::Tensor &indices,
    const std::optional<torch::stable::Tensor> &topk_length,
    const std::optional<torch::stable::Tensor> &attn_sink,
    std::optional<torch::stable::Tensor> &tile_scheduler_metadata,
    std::optional<torch::stable::Tensor> &num_splits,
    const std::optional<torch::stable::Tensor> &extra_kv,
    const std::optional<torch::stable::Tensor> &extra_indices,
    const std::optional<torch::stable::Tensor> &extra_topk_length,
    int64_t d_v,
    double sm_scale,
    // Names the layout of `kv` ("V32", "V32_NO_ROPE", "V4"). Only needed to tell V3.2-no-RoPE
    // apart from V4.1, which share 528 bytes per token; the current side never sees that
    // ambiguity because it only accepts the V4.1 formats.
    const std::optional<std::string> &kv_format);

std::tuple<torch::stable::Tensor, torch::stable::Tensor, std::optional<torch::stable::Tensor>, std::optional<torch::stable::Tensor>>
dense_attn_decode_interface(
    torch::stable::Tensor &q,
    const torch::stable::Tensor &kcache,
    const int head_size_v,
    const torch::stable::Tensor &seqlens_k,
    const torch::stable::Tensor &block_table,
    const float softmax_scale,
    bool is_causal,
    std::optional<torch::stable::Tensor> &tile_scheduler_metadata,
    std::optional<torch::stable::Tensor> &num_splits);

std::vector<torch::stable::Tensor> sparse_attn_prefill_interface(
    const torch::stable::Tensor &q,
    const torch::stable::Tensor &kv,
    const torch::stable::Tensor &indices,
    double sm_scale,
    int64_t d_v,
    const std::optional<torch::stable::Tensor> &attn_sink,
    const std::optional<torch::stable::Tensor> &topk_length);

std::vector<torch::stable::Tensor> fused_norm_rope_attn_rope_cast_fwd(
    const torch::stable::Tensor &q,
    const torch::stable::Tensor &kv,
    const torch::stable::Tensor &indices,
    double sm_scale,
    int d_v,
    const std::optional<torch::stable::Tensor> &attn_sink,
    const std::optional<torch::stable::Tensor> &topk_length,
    bool enable_q_norm,
    double rms_norm_eps,
    const torch::stable::Tensor &token_positions,
    bool is_rope_neox_style,
    uint32_t rope_dim,
    const torch::stable::Tensor &cos_sin_cache,
    uint32_t n_wv_group,
    uint32_t num_per_channels,
    bool use_tma_aligned_col_major_sf,
    bool round_sf,
    bool use_packed_ue8m0);

std::vector<torch::stable::Tensor> fused_norm_rope_attn_rope_cast_decode(
    const torch::stable::Tensor &q,
    const torch::stable::Tensor &kv,
    const torch::stable::Tensor &indices,
    double sm_scale,
    int64_t d_v,
    const std::optional<torch::stable::Tensor> &attn_sink,
    const std::optional<torch::stable::Tensor> &topk_length,
    const std::optional<torch::stable::Tensor> &extra_kv,
    const std::optional<torch::stable::Tensor> &extra_indices,
    const std::optional<torch::stable::Tensor> &extra_topk_length,
    bool enable_q_norm,
    double rms_norm_eps,
    const torch::stable::Tensor &token_positions,
    bool is_rope_neox_style,
    uint32_t rope_dim,
    const torch::stable::Tensor &cos_sin_cache,
    uint32_t n_wv_group,
    uint32_t num_per_channels,
    bool use_tma_aligned_col_major_sf,
    bool round_sf,
    bool use_packed_ue8m0);

std::vector<torch::stable::Tensor> permute_q_b_proj(
    const torch::stable::Tensor &q_b_proj,
    const torch::stable::Tensor &scale_factors,
    int h_q,
    int d_q);

std::vector<torch::stable::Tensor> permute_wv_proj(
    const torch::stable::Tensor &wv_proj,
    const torch::stable::Tensor &scale_factors,
    int wv_group_size,
    int d_o);

// --- The SGLang compatibility layer -------------------------------------------------------
// Defined in csrc_legacy/python_api.cpp and csrc_legacy/extension/sm90/dense_fp8/. These never
// existed upstream: the fork added them so that sgl-kernel could register ops with the
// int64/double types a torch schema requires. Their fate is decided by the SGLang migration.

std::vector<torch::stable::Tensor> get_mla_decoding_metadata(
    torch::stable::Tensor &seqlens_k,
    const int64_t num_q_tokens_per_head_k,
    const int64_t h_k,
    const std::optional<int64_t> h_q,
    const bool is_fp8_kvcache,
    const std::optional<int64_t> topk);

std::vector<torch::stable::Tensor> fwd_kvcache_mla(
    torch::stable::Tensor &q,
    const torch::stable::Tensor &kcache,
    const int64_t head_size_v,
    const torch::stable::Tensor &seqlens_k,
    const torch::stable::Tensor &block_table,
    const double softmax_scale,
    bool is_causal,
    const torch::stable::Tensor &tile_scheduler_metadata,
    const torch::stable::Tensor &num_splits,
    const bool &is_fp8,
    const std::optional<torch::stable::Tensor> &indices,
    const std::optional<torch::stable::Tensor> &attn_sink = std::nullopt,
    const std::optional<torch::stable::Tensor> &extra_k_cache = std::nullopt,
    const std::optional<torch::stable::Tensor> &extra_indices_in_kvcache = std::nullopt,
    const std::optional<torch::stable::Tensor> &topk_length = std::nullopt,
    const std::optional<torch::stable::Tensor> &extra_topk_length = std::nullopt);

std::vector<torch::stable::Tensor> sparse_prefill_fwd(
    const torch::stable::Tensor &q,
    const torch::stable::Tensor &kv,
    const torch::stable::Tensor &indices,
    double sm_scale,
    int64_t d_v,
    const std::optional<torch::stable::Tensor> &attn_sink = std::nullopt,
    const std::optional<torch::stable::Tensor> &topk_length = std::nullopt);

std::vector<torch::stable::Tensor> fwd_kvcache_mla_fp8(
    torch::stable::Tensor &q,
    const torch::stable::Tensor &kcache,
    const int64_t head_size_v,
    const torch::stable::Tensor &seqlens_k,
    const torch::stable::Tensor &block_table,
    const double softmax_scale,
    bool is_causal,
    const torch::stable::Tensor &tile_scheduler_metadata,
    const torch::stable::Tensor &num_splits,
    const std::optional<torch::stable::Tensor> &descale_q,
    const std::optional<torch::stable::Tensor> &descale_k);

std::vector<torch::stable::Tensor> get_mla_decoding_metadata_dense_fp8(
    torch::stable::Tensor &seqlens_k,
    const int64_t num_heads_per_head_k,
    const int64_t num_heads_k);
