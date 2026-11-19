#pragma once

#include "kernels/params.h"

// Sparse prefill with an extra KV cache: one launch attends to the tokens selected from both caches with a
// single online softmax. See SparseAttnExtraKvFwdParams in kernels/params.h for the contract.
namespace legacy::sm100::prefill::sparse_fwd::extra_kv {

// 8 or 32 active heads (params.active_heads) with Q padded to 64 heads; only active heads are written
namespace small_heads {
template<SparseAttnFwdMode FWD_MODE, int D_QK>
void run_sparse_fwd_phase1_kernel(const SparseAttnExtraKvFwdParams& params);
}

// All 64 heads
namespace head64 {
template<SparseAttnFwdMode FWD_MODE, int D_QK>
void run_sparse_fwd_phase1_kernel(const SparseAttnExtraKvFwdParams& params);
}

}
