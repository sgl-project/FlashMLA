#include "../phase1_small_heads.cuh"

namespace legacy::sm100::prefill::sparse_fwd::extra_kv::small_heads {

template void run_sparse_fwd_phase1_kernel<SparseAttnFwdMode::Prefill, 576>(const SparseAttnExtraKvFwdParams& params);

}
