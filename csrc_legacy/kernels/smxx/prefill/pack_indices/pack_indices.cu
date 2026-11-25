#include "pack_indices.h"

#include <cstdio>
#include <cstdlib>
#include <utility>

#include <cuda_runtime.h>

#include "kernels/utils.h"

/*
Index packing for token-packed sparse attention.

One CTA per group of `group` consecutive query rows. It writes the union of their top-k slots, each slot once and
tagged in bits 26..30 with the rows of the group that selected it (SPARSE_DECODE_GROUP_SHIFT, the format sparse decode
and the token-packing prefill kernel read).

The result is deterministic:
- Phase 1 inserts the slots into a hash table by prioritized linear probing: a larger slot id displaces a smaller one
  along the probe sequence. The final layout of such a table does not depend on the insertion order (Shun & Blelloch,
  "Phase-concurrent hash tables for determinism", SPAA 2014).
- Phase 2 ORs the row bits into the entries, which no longer move.
- The output is the table's non-empty entries in table order.

Only the tail of the last 64-entry block is filled with -1 (and the first two blocks, which the attention kernel
always reads), not the whole output row; the number of distinct slots goes to `lengths`.
*/

namespace legacy::smxx::prefill {

namespace {

static constexpr int TABLE_SIZE = 16384;
static constexpr int NUM_THREADS = 256;
static constexpr int ENTRIES_PER_THREAD = TABLE_SIZE / NUM_THREADS;
static constexpr int SLOT_MASK = (1 << SPARSE_DECODE_GROUP_SHIFT) - 1;
static_assert(SPARSE_PACK_INDICES_MAX_ENTRIES <= TABLE_SIZE / 4 * 3);

__device__ __forceinline__ unsigned hash_slot(int slot) {
    return ((unsigned)slot * 2654435761u) >> 18;    // 14 bits: TABLE_SIZE entries
}

__global__ void __launch_bounds__(NUM_THREADS)
sparse_pack_indices_kernel(const SparsePackIndicesParams params) {
    extern __shared__ int table[];
    __shared__ int counts[NUM_THREADS + 1];
    const int tid = threadIdx.x;
    const int64_t first_row = (int64_t)blockIdx.x * params.group;
    for (int i = tid; i < TABLE_SIZE; i += NUM_THREADS) table[i] = -1;
    __syncthreads();
    const int num_entries = params.group * params.topk;
    // Phase 1: the slots, in an order-independent layout
    for (int e = tid; e < num_entries; e += NUM_THREADS) {
        int v = params.indices[(first_row + e / params.topk) * params.stride_indices + e % params.topk];
        if (v < 0) continue;
        v &= SLOT_MASK;
        unsigned i = hash_slot(v);
        while (true) {
            int c = table[i];
            if (c == v) break;
            if (c > v) { i = (i + 1) & (TABLE_SIZE - 1); continue; }
            int prev = atomicCAS(&table[i], c, v);
            if (prev != c) continue;          // Lost the race: look at this entry again
            if (c == -1) break;
            v = c;                            // Displaced a lower-priority slot: re-insert it further on
            i = (i + 1) & (TABLE_SIZE - 1);
        }
    }
    __syncthreads();
    // Phase 2: which rows of the group selected each slot
    for (int e = tid; e < num_entries; e += NUM_THREADS) {
        int v = params.indices[(first_row + e / params.topk) * params.stride_indices + e % params.topk];
        if (v < 0) continue;
        v &= SLOT_MASK;
        unsigned i = hash_slot(v);
        while ((table[i] & SLOT_MASK) != v) i = (i + 1) & (TABLE_SIZE - 1);
        atomicOr(&table[i], 1 << (SPARSE_DECODE_GROUP_SHIFT + e / params.topk));
    }
    __syncthreads();
    // Compaction in table order
    int vals[ENTRIES_PER_THREAD], cnt = 0;
    #pragma unroll
    for (int j = 0; j < ENTRIES_PER_THREAD; ++j) {
        vals[j] = table[tid * ENTRIES_PER_THREAD + j];
        cnt += vals[j] != -1;
    }
    counts[tid + 1] = cnt;
    __syncthreads();
    if (tid == 0) {
        counts[0] = 0;
        for (int i = 1; i <= NUM_THREADS; ++i) counts[i] += counts[i - 1];
    }
    __syncthreads();
    int* dst = params.packed + (int64_t)blockIdx.x * params.stride_packed;
    int off = counts[tid];
    const int total = counts[NUM_THREADS];
    #pragma unroll
    for (int j = 0; j < ENTRIES_PER_THREAD; ++j)
        if (vals[j] != -1) dst[off++] = vals[j];
    int pad_end = (total + 63) / 64 * 64;
    if (pad_end < 128) pad_end = 128;
    for (int i = total + tid; i < pad_end; i += NUM_THREADS) dst[i] = -1;
    if (tid == 0) params.lengths[blockIdx.x] = total;
}

}  // namespace

void run_sparse_pack_indices_kernel(const SparsePackIndicesParams &params) {
    FLASH_ASSERT(params.group >= 1 && params.group <= SPARSE_DECODE_MAX_GROUPS);
    FLASH_ASSERT(params.topk >= 1 && params.group * params.topk <= SPARSE_PACK_INDICES_MAX_ENTRIES);
    constexpr size_t smem_size = TABLE_SIZE * sizeof(int);
    CHECK_CUDA(cudaFuncSetAttribute(sparse_pack_indices_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    sparse_pack_indices_kernel<<<params.num_groups, NUM_THREADS, smem_size, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}
