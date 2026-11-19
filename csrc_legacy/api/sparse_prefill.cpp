#include "common.h"
#include "kernels/sm100/prefill/sparse/fwd/small_heads/phase1.h"
#include "kernels/sm100/prefill/sparse/fwd/extra_kv/phase1.h"

#include "kernels/params.h"

#include "kernels/sm90/prefill/sparse/phase1.h"
#include "kernels/sm100/prefill/sparse/fwd/head128/phase1.h"
#include "kernels/sm100/prefill/sparse/fwd/head64/phase1.h"
#include "kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/phase1.h"

enum class FwdFeatures : int {
    HEAD_64,
    HEAD_128,

    HEAD_DIM_576,
    HEAD_DIM_512,

    ATTN_SINK,
    TOPK_LENGTH
};

class FwdImplBase : public ImplBase<
    SparseAttnFwdParams,
    FwdFeatures
> {};

class Fwd_Sm90_Impl : public FwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        FwdFeatures::HEAD_64,
        FwdFeatures::HEAD_128,
        FwdFeatures::HEAD_DIM_512,
        FwdFeatures::HEAD_DIM_576,
        FwdFeatures::ATTN_SINK,
        FwdFeatures::TOPK_LENGTH
    )

protected:
    void run_(const SparseAttnFwdParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&]() {
            DISPATCH_BOOLEAN_FLAG(params.topk_length != nullptr, HAVE_TOPK_LENGTH, [&]() {
                legacy::sm90::prefill::sparse_fwd::run_fwd_phase1_kernel<HEAD_DIM_QK, HAVE_TOPK_LENGTH>(params);
            });
        });
    }
};

class Fwd_Sm100_Head64_Impl : public FwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        FwdFeatures::HEAD_64,
        FwdFeatures::HEAD_DIM_512,
        FwdFeatures::HEAD_DIM_576,
        FwdFeatures::ATTN_SINK,
        FwdFeatures::TOPK_LENGTH
    )

protected:
    void run_(const SparseAttnFwdParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&]() {
            STD_TORCH_CHECK(params.h_q == 64, "Unsupported h_q: ", params.h_q);
            legacy::sm100::prefill::sparse_fwd::head64::run_sparse_fwd_phase1_kernel<SparseAttnFwdMode::Prefill, HEAD_DIM_QK>(params);
        });
    }
};

class Fwd_Sm100_Head128_Impl : public FwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        FwdFeatures::HEAD_128,
        FwdFeatures::HEAD_DIM_512,
        FwdFeatures::HEAD_DIM_576,
        FwdFeatures::ATTN_SINK,
        FwdFeatures::TOPK_LENGTH
    )

protected:
    void run_(const SparseAttnFwdParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&]() {
            legacy::sm100::prefill::sparse_fwd::head128::run_sparse_fwd_phase1_kernel<HEAD_DIM_QK>(params);
        });
    }
};

class Fwd_Sm100_Head128_Small_TopK_Impl : public FwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        FwdFeatures::HEAD_128,
        FwdFeatures::HEAD_DIM_512,
        FwdFeatures::ATTN_SINK,
        FwdFeatures::TOPK_LENGTH
    )

protected:
    void run_(const SparseAttnFwdParams &params, const std::vector<FeatureT> &required_features) override {
        legacy::sm100::prefill::sparse_fwd_for_small_topk::head128::run_sparse_fwd_for_small_topk_phase1_kernel<SparseAttnFwdMode::Prefill, 512>(params);
    }
};

std::vector<std::optional<Tensor>> sparse_attn_prefill_impl(
    const Tensor &q,
    const Tensor &kv,
    const Tensor &indices,
    double sm_scale,
    int64_t d_v,
    const std::optional<Tensor> &attn_sink,
    const std::optional<Tensor> &topk_length,
    bool return_log2_stats = false,
    bool return_stats = true,
    int64_t active_heads = 0,
    const std::optional<Tensor> &extra_kv = std::nullopt,
    const std::optional<Tensor> &extra_indices = std::nullopt,
    const std::optional<Tensor> &extra_topk_length = std::nullopt
) {
    using bf16 = cutlass::bfloat16_t;
    
    Arch arch = Arch();
    bool is_sm90a = arch.is_sm90a();
    bool is_sm100f = arch.is_sm100f();
    STD_TORCH_CHECK(is_sm90a || is_sm100f, "Sparse Attention Forward Kernel is only supported on SM90a and SM100f architectures.");

    KU_CHECK_NDIM(q, 3);
    KU_CHECK_NDIM(kv, 3);
    KU_CHECK_NDIM(indices, 3);
    KU_CHECK_NDIM(attn_sink, 1);
    KU_CHECK_NDIM(topk_length, 1);

    int s_q = q.size(0);
    int s_kv = kv.size(0);
    int h_q = q.size(1);
    int h_kv = kv.size(1);
    int d_qk = q.size(2);
    int topk = indices.size(2);
    bool have_topk_length = topk_length.has_value();

    STD_TORCH_CHECK(d_qk == 576 || d_qk == 512, "Invalid d_qk: ", d_qk);
    STD_TORCH_CHECK(d_v == 512, "Invalid d_v", d_v);
    
    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(topk_length);
    
    KU_CHECK_DTYPE(q, ScalarType::BFloat16);
    KU_CHECK_DTYPE(kv, ScalarType::BFloat16);
    KU_CHECK_DTYPE(indices, ScalarType::Int);
    KU_CHECK_DTYPE(attn_sink, ScalarType::Float);
    KU_CHECK_DTYPE(topk_length, ScalarType::Int);
    
    KU_CHECK_SHAPE(q, s_q, h_q, d_qk);
    KU_CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    KU_CHECK_SHAPE(indices, s_q, h_kv, topk);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(topk_length, s_q);
    
    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_LAST_DIM_CONTIGUOUS(attn_sink);
    KU_CHECK_LAST_DIM_CONTIGUOUS(topk_length);

    // Extra KV cache: each query also attends to the tokens extra_indices selects from extra_kv, in the same softmax
    bool have_extra_kv = extra_kv.has_value();
    STD_TORCH_CHECK(have_extra_kv == extra_indices.has_value(), "extra_kv and extra_indices must be given together");
    STD_TORCH_CHECK(have_extra_kv || !extra_topk_length.has_value(), "extra_topk_length requires extra_kv");
    int extra_s_kv = 0, extra_topk = 0;
    if (have_extra_kv) {
        STD_TORCH_CHECK(is_sm100f, "sparse MLA prefill with an extra KV cache requires SM10x");
        STD_TORCH_CHECK(h_q == 64 && h_kv == 1 && d_qk == 576,
            "sparse MLA prefill with an extra KV cache requires Q padded to 64 heads, one KV head and d_qk 576");
        KU_CHECK_NDIM(extra_kv, 3);
        KU_CHECK_NDIM(extra_indices, 3);
        KU_CHECK_NDIM(extra_topk_length, 1);
        extra_s_kv = extra_kv->size(0);
        extra_topk = extra_indices->size(2);
        STD_TORCH_CHECK(topk >= 128 && topk % 64 == 0 && extra_topk >= 128 && extra_topk % 64 == 0,
            "with an extra KV cache, topk and extra_topk must be >= 128 and divisible by 64, got ", topk, " and ", extra_topk);
        KU_CHECK_DEVICE(extra_kv);
        KU_CHECK_DEVICE(extra_indices);
        KU_CHECK_DEVICE(extra_topk_length);
        for (const std::optional<Tensor> *t : {&extra_kv, &extra_indices, &extra_topk_length}) {
            STD_TORCH_CHECK(!t->has_value() || (*t)->get_device_index() == q.get_device_index(),
                "extra_kv, extra_indices and extra_topk_length must be on Q's device");
        }
        KU_CHECK_DTYPE(extra_kv, ScalarType::BFloat16);
        KU_CHECK_DTYPE(extra_indices, ScalarType::Int);
        KU_CHECK_DTYPE(extra_topk_length, ScalarType::Int);
        KU_CHECK_SHAPE(extra_kv, extra_s_kv, 1, d_qk);
        KU_CHECK_SHAPE(extra_indices, s_q, 1, extra_topk);
        KU_CHECK_SHAPE(extra_topk_length, s_q);
        KU_CHECK_LAST_DIM_CONTIGUOUS(extra_kv);
        KU_CHECK_LAST_DIM_CONTIGUOUS(extra_indices);
        KU_CHECK_LAST_DIM_CONTIGUOUS(extra_topk_length);
    }

    // Allocate results and buffers
    torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());
    
    if (active_heads) {
        STD_TORCH_CHECK(is_sm100f && h_q == 64 && h_kv == 1 &&
            (active_heads == 8 || active_heads == 32) && topk >= 128 && topk % 64 == 0,
            "small-head sparse MLA prefill requires SM10x, Q64, KV1, active heads 8/32 and topk >=128 divisible by 64");
        STD_TORCH_CHECK(!return_stats, "small-head sparse MLA prefill is output-only");
    }
    Tensor out = torch::stable::new_empty(q, {s_q, active_heads ? active_heads : h_q, d_v});
    std::optional<Tensor> lse, max_logits;
    if (return_stats) {
        lse = torch::stable::new_empty(q, {s_q, h_q}, ScalarType::Float);
        max_logits = torch::stable::new_empty(q, {s_q, h_q}, ScalarType::Float);
        KU_CHECK_CONTIGUOUS(lse);
        KU_CHECK_CONTIGUOUS(max_logits);
    }
    KU_CHECK_CONTIGUOUS(out);

    // All prefill kernels emit the requested units in their final stores.
    // Null stats pointers select output-only mode without extra allocations.
    SparseAttnFwdParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, (int)d_v, topk,
        (float)sm_scale, (float)(sm_scale * LOG_2_E),

        (bf16*)q.data_ptr(),
        (bf16*)kv.data_ptr(),
        (int*)indices.data_ptr(),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        ku::get_optional_tensor_ptr<int>(topk_length),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        (bf16*)out.data_ptr(),
        ku::get_optional_tensor_ptr<float>(max_logits),
        ku::get_optional_tensor_ptr<float>(lse),

        arch.num_sms,
        return_log2_stats,
        get_current_stream()
    };

    if (have_extra_kv) {
        SparseAttnExtraKvFwdParams extra_params{};
        static_cast<SparseAttnFwdParams&>(extra_params) = params;
        extra_params.active_heads = active_heads ? active_heads : h_q;
        extra_params.output_heads = extra_params.active_heads;
        extra_params.extra_kv = (bf16*)extra_kv->data_ptr();
        extra_params.extra_indices = (int*)extra_indices->data_ptr();
        extra_params.extra_topk_length = ku::get_optional_tensor_ptr<int>(extra_topk_length);
        extra_params.extra_s_kv = extra_s_kv;
        extra_params.extra_topk = extra_topk;
        extra_params.stride_extra_kv_s_kv = int64_stride_to_int(extra_kv->stride(0));
        extra_params.stride_extra_indices_s_q = int64_stride_to_int(extra_indices->stride(0));
        if (s_q) {
            if (active_heads) {
                legacy::sm100::prefill::sparse_fwd::extra_kv::small_heads::run_sparse_fwd_phase1_kernel<SparseAttnFwdMode::Prefill, 576>(extra_params);
            } else {
                legacy::sm100::prefill::sparse_fwd::extra_kv::head64::run_sparse_fwd_phase1_kernel<SparseAttnFwdMode::Prefill, 576>(extra_params);
            }
        }
        return {out, max_logits, lse};
    }

    if (active_heads) {
        SparseAttnSmallHeadsFwdParams small_heads_params{};
        static_cast<SparseAttnFwdParams&>(small_heads_params) = params;
        small_heads_params.active_heads = active_heads;
        small_heads_params.output_heads = active_heads;
        if (s_q) {
            DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&]() {
                legacy::sm100::prefill::sparse_fwd::small_heads::run_sparse_fwd_phase1_kernel<SparseAttnFwdMode::Prefill, HEAD_DIM_QK>(small_heads_params);
            });
        }
        return {out, max_logits, lse};
    }

    std::vector<FwdFeatures> required_features;
    if (h_q == 64) {
        required_features.push_back(FwdFeatures::HEAD_64);
    } else if (h_q == 128) {
        required_features.push_back(FwdFeatures::HEAD_128);
    } else {
        STD_TORCH_CHECK(false, "Unsupported h_q: ", h_q);
    }
    if (d_qk == 576) {
        required_features.push_back(FwdFeatures::HEAD_DIM_576);
    } else if (d_qk == 512) {
        required_features.push_back(FwdFeatures::HEAD_DIM_512);
    } else {
        STD_TORCH_CHECK(false, "Unsupported d_qk: ", d_qk);
    }
    if (attn_sink.has_value()) {
        required_features.push_back(FwdFeatures::ATTN_SINK);
    }
    if (have_topk_length) {
        required_features.push_back(FwdFeatures::TOPK_LENGTH);
    }

    if (is_sm90a) {
        Fwd_Sm90_Impl fwd_impl;
        fwd_impl.run(params, required_features);
    } else if (is_sm100f) {
        if (h_q == 64) {
            Fwd_Sm100_Head64_Impl fwd_impl;
            fwd_impl.run(params, required_features);
        } else if (h_q == 128) {
            Fwd_Sm100_Head128_Small_TopK_Impl small_topk_impl;
            Fwd_Sm100_Head128_Impl regular_impl;
            bool use_small_topk_impl = false;
            if (
                (topk <= 1280 && small_topk_impl.check_if_all_features_are_supported(required_features)) ||
                !regular_impl.check_if_all_features_are_supported(required_features)
            ) {
                use_small_topk_impl = true;
            }
            if (use_small_topk_impl) {
                small_topk_impl.run(params, required_features);
            } else {
                regular_impl.run(params, required_features);
            }
        } else {
            STD_TORCH_CHECK(false, "Unsupported h_q: ", h_q);
        }
    } else {
        STD_TORCH_CHECK(false, "Unsupported architecture");
    }

    return {out, max_logits, lse};
}

std::vector<std::optional<Tensor>> sparse_attn_prefill_interface(
    const Tensor &q, const Tensor &kv, const Tensor &indices, double sm_scale,
    int64_t d_v, const std::optional<Tensor> &attn_sink,
    const std::optional<Tensor> &topk_length, bool return_log2_stats, bool return_stats,
    const std::optional<Tensor> &extra_kv, const std::optional<Tensor> &extra_indices,
    const std::optional<Tensor> &extra_topk_length) {
    return sparse_attn_prefill_impl(q, kv, indices, sm_scale, d_v, attn_sink,
        topk_length, return_log2_stats, return_stats, 0, extra_kv, extra_indices, extra_topk_length);
}

std::vector<std::optional<Tensor>> sparse_prefill_fwd_small_heads(
    const Tensor &q, const Tensor &kv, const Tensor &indices, double sm_scale,
    int64_t active_heads, const std::optional<Tensor> &attn_sink,
    const std::optional<Tensor> &topk_length, const std::optional<Tensor> &extra_kv,
    const std::optional<Tensor> &extra_indices, const std::optional<Tensor> &extra_topk_length) {
    STD_TORCH_CHECK(active_heads == 8 || active_heads == 32, "active_heads must be 8 or 32");
    return sparse_attn_prefill_impl(q, kv, indices, sm_scale, 512, attn_sink,
        topk_length, false, false, active_heads, extra_kv, extra_indices, extra_topk_length);
}

STABLE_TORCH_LIBRARY_IMPL(flash_mla_legacy, CUDA, m) {
    m.impl("sparse_prefill_fwd_small_heads", TORCH_BOX(&sparse_prefill_fwd_small_heads));
    m.impl("sparse_prefill_fwd", TORCH_BOX(&sparse_attn_prefill_interface));
}
