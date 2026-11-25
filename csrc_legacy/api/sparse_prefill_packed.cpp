#include "common.h"

#include <algorithm>

#include "kernels/params.h"
#include "kernels/sm100/prefill/sparse/fwd/small_heads/phase1.h"
#include "kernels/sm100/prefill/sparse/fwd/token_packing/phase1.h"
#include "kernels/smxx/prefill/dequant_v32/dequant_v32.h"
#include "kernels/smxx/prefill/pack_indices/pack_indices.h"

void sparse_prefill_fwd_packed(
    const Tensor &q,
    const Tensor &kv,
    const Tensor &indices,
    double sm_scale,
    int64_t heads_per_group,
    const std::optional<Tensor> &topk_length,
    const Tensor &out
) {
    using bf16 = cutlass::bfloat16_t;

    Arch arch = Arch();
    STD_TORCH_CHECK(arch.is_sm100f(), "sparse MLA prefill with token packing requires SM10x");

    KU_CHECK_NDIM(q, 3);
    KU_CHECK_NDIM(kv, 3);
    KU_CHECK_NDIM(indices, 3);
    KU_CHECK_NDIM(topk_length, 1);
    KU_CHECK_NDIM(out, 3);

    int s_q = q.size(0);
    int s_kv = kv.size(0);
    int h_q = q.size(1);
    int h_kv = kv.size(1);
    int d_qk = q.size(2);
    int topk = indices.size(2);

    STD_TORCH_CHECK(d_qk == 576, "sparse MLA prefill with token packing requires d_qk 576, got ", d_qk);
    STD_TORCH_CHECK(h_q >= 1 && h_q <= 64, "h_q must be in [1, 64], got ", h_q);
    STD_TORCH_CHECK(h_kv == 1 && topk >= 128 && topk % 64 == 0,
        "sparse MLA prefill with token packing requires one KV head and topk >= 128 divisible by 64");
    STD_TORCH_CHECK(heads_per_group >= 0, "heads_per_group must be >= 0");
    if (heads_per_group > 0) {
        STD_TORCH_CHECK(h_q <= heads_per_group * SPARSE_DECODE_MAX_GROUPS, "h_q exceeds ", SPARSE_DECODE_MAX_GROUPS,
                        " groups of heads_per_group heads");
        STD_TORCH_CHECK(s_kv <= (1 << SPARSE_DECODE_GROUP_SHIFT),
                        "token packing keeps the token-group bits above bit ", SPARSE_DECODE_GROUP_SHIFT, " of an index: s_kv is too large");
    }

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DEVICE(out);

    KU_CHECK_DTYPE(q, ScalarType::BFloat16);
    KU_CHECK_DTYPE(kv, ScalarType::BFloat16);
    KU_CHECK_DTYPE(indices, ScalarType::Int);
    KU_CHECK_DTYPE(topk_length, ScalarType::Int);
    KU_CHECK_DTYPE(out, ScalarType::BFloat16);

    KU_CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    KU_CHECK_SHAPE(indices, s_q, h_kv, topk);
    KU_CHECK_SHAPE(topk_length, s_q);
    KU_CHECK_SHAPE(out, s_q, h_q, 512);

    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_LAST_DIM_CONTIGUOUS(topk_length);
    KU_CHECK_CONTIGUOUS(out);

    torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());

    SparseAttnFwdParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, 512, topk,
        (float)sm_scale, (float)(sm_scale * LOG_2_E),

        (bf16*)q.data_ptr(),
        (bf16*)kv.data_ptr(),
        (int*)indices.data_ptr(),
        nullptr,
        ku::get_optional_tensor_ptr<int>(topk_length),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        (bf16*)out.data_ptr(),
        nullptr,
        nullptr,

        arch.num_sms,
        false,
        get_current_stream()
    };

    if (s_q == 0) {
        return;
    }
    if (heads_per_group == 0 && h_q <= 32) {
        SparseAttnSmallHeadsFwdParams small_heads_params{};
        static_cast<SparseAttnFwdParams&>(small_heads_params) = params;
        small_heads_params.active_heads = h_q;
        small_heads_params.output_heads = h_q;
        legacy::sm100::prefill::sparse_fwd::small_heads::run_sparse_fwd_phase1_kernel<SparseAttnFwdMode::Prefill, 576>(small_heads_params);
    } else {
        SparseAttnPackedFwdParams packed_params{};
        static_cast<SparseAttnFwdParams&>(packed_params) = params;
        packed_params.heads_per_group = (int)heads_per_group;
        legacy::sm100::prefill::sparse_fwd::token_packing::head64::run_sparse_fwd_phase1_kernel<SparseAttnFwdMode::Prefill, 576>(packed_params);
    }
}

void sparse_prefill_pack_indices(const Tensor &indices, int64_t group, const Tensor &packed, const Tensor &lengths) {
    KU_CHECK_NDIM(indices, 2);
    KU_CHECK_NDIM(packed, 2);
    KU_CHECK_NDIM(lengths, 1);

    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(packed);
    KU_CHECK_DEVICE(lengths);

    KU_CHECK_DTYPE(indices, ScalarType::Int);
    KU_CHECK_DTYPE(packed, ScalarType::Int);
    KU_CHECK_DTYPE(lengths, ScalarType::Int);

    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_CONTIGUOUS(packed);
    KU_CHECK_CONTIGUOUS(lengths);

    int num_rows = indices.size(0), topk = indices.size(1);
    STD_TORCH_CHECK(group >= 1 && group <= SPARSE_DECODE_MAX_GROUPS && num_rows % group == 0,
        "group must be in [1, ", SPARSE_DECODE_MAX_GROUPS, "] and divide the number of index rows");
    STD_TORCH_CHECK(topk >= 1 && topk * group <= legacy::smxx::prefill::SPARSE_PACK_INDICES_MAX_ENTRIES,
        "group * topk must be in [1, ", legacy::smxx::prefill::SPARSE_PACK_INDICES_MAX_ENTRIES, "]");
    int num_groups = num_rows / (int)group;
    STD_TORCH_CHECK(packed.size(0) >= num_groups && packed.size(1) % 64 == 0 &&
                    packed.size(1) >= std::max((int64_t)128, (group * topk + 63) / 64 * 64),
        "packed must hold a row per group, of a width divisible by 64 and at least max(128, group * topk rounded up to 64)");
    STD_TORCH_CHECK(lengths.size(0) >= num_groups, "lengths must hold an entry per group");

    torch::stable::accelerator::DeviceGuard device_guard(indices.get_device_index());

    if (num_groups == 0) {
        return;
    }
    SparsePackIndicesParams params = {
        (const int*)indices.data_ptr(), indices.stride(0),
        topk, (int)group,
        (int*)packed.data_ptr(), int64_stride_to_int(packed.stride(0)),
        (int*)lengths.data_ptr(),
        num_groups,
        get_current_stream()
    };
    legacy::smxx::prefill::run_sparse_pack_indices_kernel(params);
}

void dequant_v32_kv(const Tensor &kv, const std::optional<Tensor> &rows, const Tensor &out) {
    KU_CHECK_NDIM(kv, 4);
    KU_CHECK_NDIM(rows, 1);
    KU_CHECK_NDIM(out, 3);

    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(rows);
    KU_CHECK_DEVICE(out);

    KU_CHECK_DTYPE(rows, ScalarType::Int);
    KU_CHECK_DTYPE(out, ScalarType::BFloat16);

    KU_CHECK_CONTIGUOUS(rows);
    KU_CHECK_CONTIGUOUS(out);

    STD_TORCH_CHECK(kv.size(1) == 64 && kv.size(2) == 1 && kv.size(3) == 656 && kv.stride(1) == 656,
        "kv must be V3.2 FP8 pages [num_pages, 64, 1, 656] with contiguous pages");
    STD_TORCH_CHECK(out.size(1) == 1 && out.size(2) == 576, "out must be [n, 1, 576]");
    int64_t n = out.size(0);
    if (rows.has_value()) {
        STD_TORCH_CHECK(rows->size(0) >= n, "rows must hold a slot id per output row");
    } else {
        STD_TORCH_CHECK(n <= kv.size(0) * 64, "out has more rows than kv has slots");
    }

    torch::stable::accelerator::DeviceGuard device_guard(kv.get_device_index());

    if (n == 0) {
        return;
    }
    DequantV32Params params = {
        (const uint8_t*)kv.data_ptr(), kv.stride(0),
        rows.has_value() ? (const int*)rows->data_ptr() : nullptr,
        (uint16_t*)out.data_ptr(),
        n,
        get_current_stream()
    };
    legacy::smxx::prefill::run_dequant_v32_kernel(params);
}

STABLE_TORCH_LIBRARY_IMPL(flash_mla_legacy, CUDA, m) {
    m.impl("sparse_prefill_fwd_packed", TORCH_BOX(&sparse_prefill_fwd_packed));
    m.impl("sparse_prefill_pack_indices", TORCH_BOX(&sparse_prefill_pack_indices));
    m.impl("dequant_v32_kv", TORCH_BOX(&dequant_v32_kv));
}
