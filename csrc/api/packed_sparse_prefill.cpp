#include "common.h"
#include "kernels/params.h"
#include "kernels/sm100/prefill/sparse/fwd/head64/phase1.h"

void prepare_packed_sparse_prefill(const at::Tensor &, const at::Tensor &, int, at::Tensor &,
                                   at::Tensor &, at::Tensor &);

// Explicit 16-head API: a padded 64-head tensor does not establish which heads
// are semantically live. Strided slices of a padded allocation are supported.
std::vector<at::Tensor>
packed_sparse_attn_prefill_interface(const at::Tensor &q, const at::Tensor &kv,
                                     const at::Tensor &indices, const at::Tensor &lengths,
                                     float sm_scale, const std::optional<at::Tensor> &attn_sink) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    at::cuda::CUDAGuard guard(q.device());
    const Arch arch;
    TORCH_CHECK(arch.major == 10 && arch.minor == 0 && arch.num_sms == 148,
                "Packed BF16 sparse prefill is qualified on 148-SM B200 only");
    TORCH_CHECK(q.dim() == 3 && q.size(1) == 16 && q.size(2) == 512,
                "q must have shape [tokens, 16, 512]");
    TORCH_CHECK(kv.dim() == 3 && kv.size(1) == 1 && kv.size(2) == 512,
                "kv must have shape [history, 1, 512]");
    TORCH_CHECK(indices.dim() == 2 && indices.size(0) == q.size(0) &&
                    (indices.size(1) == 128 || indices.size(1) == 640),
                "indices must have shape [tokens, 128 or 640]");
    TORCH_CHECK(lengths.dim() == 1 && lengths.size(0) == q.size(0),
                "lengths must have shape [tokens]");
    TORCH_CHECK(q.size(0) > 0 && q.size(0) <= std::numeric_limits<int>::max() - 3 &&
                    kv.size(0) > 0 && kv.size(0) <= std::numeric_limits<int>::max(),
                "Nonempty int32-sized query and KV dimensions required");
    for (const auto &tensor : {kv, indices, lengths})
        TORCH_CHECK(tensor.device() == q.device(), "All inputs must share a CUDA device");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && kv.scalar_type() == at::kBFloat16 &&
                    indices.scalar_type() == at::kInt && lengths.scalar_type() == at::kInt,
                "Expected BF16 q/kv and int32 indices/lengths");
    TORCH_CHECK(q.stride(1) == 512 && q.stride(2) == 1 && kv.stride(2) == 1 &&
                    q.stride(0) % 8 == 0 && kv.stride(0) % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0 &&
                    indices.is_contiguous() && lengths.is_contiguous(),
                "Unsupported strides or TMA alignment");
    if (attn_sink) {
        TORCH_CHECK(attn_sink->device() == q.device() && attn_sink->dim() == 1 &&
                        attn_sink->numel() == 16 && attn_sink->scalar_type() == at::kFloat &&
                        attn_sink->is_contiguous(),
                    "sink must be contiguous float32 [16]");
    }

    const int rows = q.size(0), groups = (rows + 3) / 4;
    const int width = indices.size(1) * 4;
    auto combined = at::empty({groups, 1, width}, indices.options());
    auto combined_lengths = at::empty({groups}, lengths.options());
    auto masks = at::empty({groups, width / 64, 4}, indices.options().dtype(at::kLong));
    auto out = at::empty({groups, 64, 512}, q.options());
    auto max_logits = at::empty({groups, 64}, q.options().dtype(at::kFloat));
    auto lse = at::empty_like(max_logits);

    SparseAttnFwdParams params = {groups,
                                  static_cast<int>(kv.size(0)),
                                  64,
                                  1,
                                  512,
                                  512,
                                  width,
                                  sm_scale,
                                  sm_scale * LOG_2_E,
                                  q.data_ptr<cutlass::bfloat16_t>(),
                                  kv.data_ptr<cutlass::bfloat16_t>(),
                                  combined.data_ptr<int>(),
                                  ku::get_optional_tensor_ptr<float>(attn_sink),
                                  combined_lengths.data_ptr<int>(),
                                  int64_stride_to_int(q.stride(0) * 4),
                                  int64_stride_to_int(q.stride(1)),
                                  int64_stride_to_int(kv.stride(0)),
                                  int64_stride_to_int(kv.stride(1)),
                                  int64_stride_to_int(combined.stride(0)),
                                  int64_stride_to_int(combined.stride(1)),
                                  out.data_ptr<cutlass::bfloat16_t>(),
                                  max_logits.data_ptr<float>(),
                                  lse.data_ptr<float>(),
                                  arch.num_sms,
                                  at::cuda::getCurrentCUDAStream().stream()};
    params.packed_mask = reinterpret_cast<const uint64_t *>(masks.data_ptr());
    params.packed_q_rows = rows;
    prepare_packed_sparse_prefill(indices, lengths, params.s_kv, combined, combined_lengths, masks);
    sm100::prefill::sparse_fwd::head64::run_sparse_fwd_phase1_kernel<SparseAttnFwdMode::Prefill,
                                                                     512, true>(params);
    // Views discard padded tokens without a gather kernel. Auxiliary statistics
    // follow the native FlashMLA natural-log convention and exclude the sink.
    return {out.view({groups * 4, 16, 512}).narrow(0, 0, rows),
            max_logits.view({groups * 4, 16}).narrow(0, 0, rows),
            lse.view({groups * 4, 16}).narrow(0, 0, rows)};
}

#ifndef FLASH_MLA_LIBTORCH_ONLY
void register_packed_sparse_prefill(pybind11::module_ &m) {
    m.def("packed_sparse_prefill_fwd", &packed_sparse_attn_prefill_interface,
          "Run packed 16-head BF16 sparse prefill");
}
#endif
