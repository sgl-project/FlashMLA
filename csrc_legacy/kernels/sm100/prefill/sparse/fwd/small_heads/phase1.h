#pragma once

#include "kernels/params.h"

namespace legacy::sm100::prefill::sparse_fwd::small_heads {

template<SparseAttnFwdMode FWD_MODE, int D_QK>
void run_sparse_fwd_phase1_kernel(const SparseAttnSmallHeadsFwdParams& params);

}
