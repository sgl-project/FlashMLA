// Op schemas of the frozen pre-2026.09.30 kernels.
//
// They live in their own namespace (`flash_mla_legacy`) so that the two extensions can be
// loaded into one process: upstream reused the same op names when it rewrote the V4.1 kernels,
// and registering both under one namespace would collide at import time. Python picks the
// namespace per call (see flash_mla/_dispatch.py), which is also what makes the 528 B/token
// ambiguity go away -- only this side has formats that need `kv_format` to disambiguate.
//
// The implementations are bound next to their definitions, in csrc_legacy/api/*.cpp and
// csrc_legacy/python_api.cpp, each with its own STABLE_TORCH_LIBRARY_IMPL block.

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY(flash_mla_legacy, m) {
    // --- Sparse attention over the non-interleaved KV formats (V3.2, V3.2-no-RoPE, V4) ------
    // `kv_format` names the layout: V3.2-no-RoPE and V4.1 are both 528 bytes per token, so the
    // shape alone cannot tell them apart.
    m.def(
        "sparse_decode_fwd(Tensor q, Tensor kv, Tensor indices, Tensor? topk_length, "
        "Tensor? attn_sink, Tensor(a!)? tile_scheduler_metadata, Tensor(b!)? num_splits, "
        "Tensor? extra_kv, Tensor? extra_indices, Tensor? extra_topk_length, int d_v, "
        "float sm_scale, str? kv_format=None) -> (Tensor, Tensor, Tensor(a)?, Tensor(b)?)");
    // `return_stats=False` skips allocating and storing max_logits/lse, and both come back
    // None; `return_log2_stats=True` has the kernel emit them in base 2 instead of natural
    // logs, which is what the SGLang call sites have always been handed.
    m.def(
        "sparse_prefill_fwd(Tensor q, Tensor kv, Tensor indices, float sm_scale, int d_v, "
        "Tensor? attn_sink=None, Tensor? topk_length=None, bool return_log2_stats=False, "
        "bool return_stats=True) -> Tensor?[]");

    // --- Dense MLA decoding: SM90 only, and deleted upstream in the 2026.09.30 release -------
    m.def(
        "dense_decode_fwd(Tensor q, Tensor kcache, int head_size_v, Tensor seqlens_k, "
        "Tensor block_table, float softmax_scale, bool is_causal, "
        "Tensor(a!)? tile_scheduler_metadata, Tensor(b!)? num_splits) "
        "-> (Tensor, Tensor, Tensor(a)?, Tensor(b)?)");

    // --- Fused norm + RoPE + attn + RoPE + cast (MegaAttention) over the V4 KV format --------
    m.def(
        "fused_norm_rope_attn_rope_cast_fwd(Tensor q, Tensor kv, Tensor indices, float sm_scale, "
        "int d_v, Tensor? attn_sink, Tensor? topk_length, bool enable_q_norm, float rms_norm_eps, "
        "Tensor token_positions, bool is_rope_neox_style, int rope_dim, Tensor cos_sin_cache, "
        "int n_wv_group, int num_per_channels, bool use_tma_aligned_col_major_sf, bool round_sf, "
        "bool use_packed_ue8m0) -> Tensor[]");
    m.def(
        "fused_norm_rope_attn_rope_cast_decode(Tensor q, Tensor kv, Tensor indices, float sm_scale, "
        "int d_v, Tensor? attn_sink, Tensor? topk_length, Tensor? extra_kv, Tensor? extra_indices, "
        "Tensor? extra_topk_length, bool enable_q_norm, float rms_norm_eps, Tensor token_positions, "
        "bool is_rope_neox_style, int rope_dim, Tensor cos_sin_cache, int n_wv_group, "
        "int num_per_channels, bool use_tma_aligned_col_major_sf, bool round_sf, "
        "bool use_packed_ue8m0) -> Tensor[]");
    m.def("permute_q_b_proj(Tensor q_b_proj, Tensor scale_factors, int h_q, int d_q) -> Tensor[]");
    m.def("permute_wv_proj(Tensor wv_proj, Tensor scale_factors, int wv_group_size, int d_o) -> Tensor[]");

    // --- The fork's own SM90 dense fp8 attention --------------------------------------------
    m.def(
        "fwd_kvcache_mla_fp8(Tensor q, Tensor kcache, int head_size_v, Tensor seqlens_k, "
        "Tensor block_table, float softmax_scale, bool is_causal, Tensor tile_scheduler_metadata, "
        "Tensor num_splits, Tensor? descale_q, Tensor? descale_k) -> Tensor[]");
    m.def(
        "get_mla_decoding_metadata_dense_fp8(Tensor seqlens_k, int num_heads_per_head_k, "
        "int num_heads_k) -> Tensor[]");
}
