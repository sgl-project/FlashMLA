#include "../phase1.h"
#include "../phase1.cuh"

namespace legacy::sm100::prefill::sparse_fwd::small_heads {

template void run_sparse_fwd_phase1_kernel<SparseAttnFwdMode::Prefill, 576>(const SparseAttnSmallHeadsFwdParams& params);

}
