#include "../phase1.h"
#include "../phase1.cuh"

namespace sm100::prefill::sparse_fwd_for_small_topk::head128 {

template void run_sparse_fwd_for_small_topk_phase1_kernel<SparseAttnFwdMode::DecodeWithSplitKV, ModelType::V32_NO_ROPE, ModelType::V32_NO_ROPE>(const SparseAttnDecodeParams& params);

}
