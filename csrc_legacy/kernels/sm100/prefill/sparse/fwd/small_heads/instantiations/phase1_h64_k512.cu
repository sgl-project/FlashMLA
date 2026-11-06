#include "../phase1.h"
#include "../phase1.cuh"

namespace legacy::sm100::prefill::sparse_fwd::small_heads {

template void run_sparse_fwd_phase1_kernel<SparseAttnFwdMode::Prefill, 512>(const SparseAttnSmallHeadsFwdParams& params);

}
