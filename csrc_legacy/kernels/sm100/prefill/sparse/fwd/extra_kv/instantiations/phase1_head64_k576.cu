#include "../phase1_head64.cuh"

namespace legacy::sm100::prefill::sparse_fwd::extra_kv::head64 {

template void run_sparse_fwd_phase1_kernel<SparseAttnFwdMode::Prefill, 576>(const SparseAttnExtraKvFwdParams& params);

}
