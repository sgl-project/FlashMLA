// Precompute the split-KV scheduling metadata, hoisted out of the decode call.
//
// `sparse_decode_fwd` allocates and fills this itself when it is handed an empty
// FlashMLASchedMeta, which is upstream's design. This entry point exists because SGLang hoists
// it out of the decode loop instead, and it is registered on both sides so that the caller's
// `use_legacy` choice also picks the scheduler: the two schedulers agree on every non-empty
// partition, but they are not the same kernel, and pairing one side's metadata with the other
// side's decode is not a contract either of them makes.
//
// Ported from the fork's csrc/python_api.cpp. The field is `block_size_n` here: upstream
// renamed it to `block_size_topk` in #229, and this tree predates that. Its twin on the current
// side is csrc_ops/sched_meta.cpp; the two differ only in namespace and in this field's name.

#include <algorithm>
#include <optional>
#include <vector>

#include <cutlass/fast_math.h>

#include "api/common.h"
#include "kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"

std::vector<Tensor> get_mla_decoding_metadata(
    const Tensor &seqlens_k,
    const int64_t num_q_tokens_per_head_k,
    const int64_t h_k,
    const std::optional<int64_t> h_q,
    const bool is_fp8_kvcache,
    const std::optional<int64_t> topk) {
    KU_CHECK_DEVICE(seqlens_k);
    KU_CHECK_CONTIGUOUS(seqlens_k);
    KU_CHECK_DTYPE(seqlens_k, ScalarType::Int);

    const int batch_size = static_cast<int>(seqlens_k.size(0));
    STD_TORCH_CHECK(batch_size > 0, "batch size must be positive");

    const int num_q_tokens_per_head_k_int = static_cast<int>(num_q_tokens_per_head_k);
    const int h_k_int = static_cast<int>(h_k);
    const std::optional<int> h_q_int =
        h_q.has_value() ? std::make_optional(static_cast<int>(*h_q)) : std::nullopt;
    const std::optional<int> topk_int =
        topk.has_value() ? std::make_optional(static_cast<int>(*topk)) : std::nullopt;

    STD_TORCH_CHECK(h_k_int > 0, "num_heads_k must be positive");
    if (topk_int.has_value()) {
        STD_TORCH_CHECK(h_q_int.has_value(), "num_heads_q must be provided when topk is provided");
        STD_TORCH_CHECK(is_fp8_kvcache, "Sparse decoding requires is_fp8_kvcache=true");
    }
    STD_TORCH_CHECK(!(is_fp8_kvcache && !topk_int.has_value()),
                    "Use get_mla_decoding_metadata_dense_fp8 for dense fp8 metadata");

    const int num_heads_q = h_q_int.value_or(h_k_int * num_q_tokens_per_head_k_int);
    STD_TORCH_CHECK(num_heads_q > 0, "num_heads_q must be positive");

    const int heads_ratio = std::max(1, num_heads_q / h_k_int);
    const int s_q = std::max(1, num_q_tokens_per_head_k_int / heads_ratio);

    Arch arch;
    int num_sm_parts;
    if (topk_int.has_value()) {
        if (arch.is_sm100f()) {
            // sm100 sparse kernels use a larger split count envelope.
            num_sm_parts = std::max(arch.num_sms / s_q, 1);
        } else {
            const int heads_per_64 = std::max(1, num_heads_q / 64);
            num_sm_parts = std::max(arch.num_sms / s_q / heads_per_64, 1);
        }
    } else {
        num_sm_parts = std::max(
            arch.num_sms / h_k_int / cutlass::ceil_div(s_q * num_heads_q / h_k_int, 64), 1);
    }

    torch::stable::accelerator::DeviceGuard device_guard(seqlens_k.get_device_index());

    Tensor tile_scheduler_metadata = torch::stable::new_empty(
        seqlens_k, {num_sm_parts, DecodingSchedMetaSize / static_cast<int>(sizeof(int))},
        ScalarType::Int);
    Tensor num_splits = torch::stable::new_empty(seqlens_k, {batch_size + 1}, ScalarType::Int);

    // Field order follows this tree's own params.h, which predates upstream's reshuffle.
    GetDecodeSchedMetaParams params = {
        .b = batch_size,
        .s_q = s_q,
        .block_size_n = 64,
        .fixed_overhead_num_blocks = 5,
        .topk = topk_int.value_or(-1),
        .extra_topk = -1,
        .topk_length = nullptr,
        .extra_topk_length = nullptr,
        .seqlens_k_ptr = const_cast<int *>(seqlens_k.const_data_ptr<int>()),
        .tile_scheduler_metadata_ptr =
            reinterpret_cast<DecodingSchedMeta *>(tile_scheduler_metadata.mutable_data_ptr<int>()),
        .num_splits_ptr = num_splits.mutable_data_ptr<int>(),
        .num_sm_parts = num_sm_parts,
        .stream = get_current_stream(),
    };

    legacy::smxx::decode::run_get_decoding_sched_meta_kernel(params);
    return {tile_scheduler_metadata, num_splits};
}

STABLE_TORCH_LIBRARY_FRAGMENT(flash_mla_legacy, m) {
    m.def(
        "get_mla_decoding_metadata(Tensor seqlens_k, int num_q_tokens_per_head_k, int h_k, "
        "int? h_q, bool is_fp8_kvcache, int? topk) -> Tensor[]");
}

STABLE_TORCH_LIBRARY_IMPL(flash_mla_legacy, CUDA, m) {
    m.impl("get_mla_decoding_metadata", TORCH_BOX(&get_mla_decoding_metadata));
}
