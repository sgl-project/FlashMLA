#pragma once

#include "kernels/params.h"

namespace legacy::smxx::prefill {

// Largest group * topk one group may hold: the hash table has 16384 entries and is kept at most 3/4 full
static constexpr int SPARSE_PACK_INDICES_MAX_ENTRIES = 12288;

void run_sparse_pack_indices_kernel(const SparsePackIndicesParams &params);

}
