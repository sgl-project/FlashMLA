#pragma once

#include "kernels/params.h"

// Sparse prefill with token packing: several query tokens share one 64-row tile and attend to the union of their
// top-k lists; bits 26..30 of an index say which of them selected it. See SparseAttnPackedFwdParams in
// kernels/params.h for the contract. Output only (no max_logits / lse, no attention sink).
namespace legacy::sm100::prefill::sparse_fwd::token_packing {

// 1 <= h_q <= 64 real rows per packed query, no padding
namespace head64 {
template<SparseAttnFwdMode FWD_MODE, int D_QK>
void run_sparse_fwd_phase1_kernel(const SparseAttnPackedFwdParams& params);
}

}
