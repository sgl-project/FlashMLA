// Build a multiset union for four query rows while retaining per-row membership.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_scan.cuh>

namespace {
constexpr int Threads = 256;

template <int Table, bool PackedKeys = false>
__global__ void prepare_kernel(const int *indices, const int *lengths, int rows, int width,
                               int history, int *combined, int *combined_lengths,
                               unsigned *row_masks) {
    extern __shared__ __align__(16) unsigned char storage[];
    unsigned *keys = reinterpret_cast<unsigned *>(storage);
    unsigned *members = reinterpret_cast<unsigned *>(keys + Table);
    // Membership storage covers every input position, including duplicates.
    unsigned char *compact_members =
        reinterpret_cast<unsigned char *>(keys + (PackedKeys ? Table : 2 * Table));
    // Valid packed entries have a nonzero membership nibble, including key zero.
    constexpr unsigned Empty = PackedKeys ? 0u : 0xffffffffu;
    __shared__ typename cub::BlockScan<int, Threads>::TempStorage scan;
    const int tid = threadIdx.x;
    const int group = blockIdx.x;
    const int output_width = width * 4;
    constexpr int Items = Table >= 4096 ? 10 : 2;
    int staged_keys[Items];
#pragma unroll
    for (int item = 0; item < Items; ++item) {
        int i = tid + item * Threads;
        int row = group * 4 + i / width, col = i % width;
        staged_keys[item] = -1;
        if (i < output_width && row < rows && col < lengths[row])
            staged_keys[item] = indices[int64_t(row) * width + col];
    }
    for (int i = tid; i < Table; i += Threads) {
        keys[i] = Empty;
        if constexpr (!PackedKeys)
            members[i] = 0;
    }
    for (int i = tid; i < output_width; i += Threads)
        combined[int64_t(group) * output_width + i] = -1;
    __syncthreads();

#pragma unroll
    for (int item = 0; item < Items; ++item) {
        int i = tid + item * Threads;
        if (i >= output_width)
            continue;
        int local_row = i / width;
        int key = -1;
        key = staged_keys[item];
        if (key < 0 || key >= history)
            continue;
        unsigned bit = 1u << local_row;
        unsigned slot = (unsigned(key) * 2654435761u) & (Table - 1);
        while (true) {
            unsigned entry = PackedKeys ? (unsigned(key) << 4) | bit : unsigned(key);
            unsigned previous = atomicCAS(keys + slot, Empty, entry);
            if constexpr (PackedKeys) {
                if (previous == Empty)
                    break;
            }
            unsigned previous_key = PackedKeys ? previous >> 4 : previous;
            if (previous == Empty || previous_key == unsigned(key)) {
                unsigned before = atomicOr((PackedKeys ? keys : members) + slot, bit);
                // A repeated position within one row gets a distinct union entry.
                if (!(before & bit))
                    break;
            }
            slot = (slot + 1) & (Table - 1);
        }
    }
    __syncthreads();
    int count = 0;
    for (int i = tid; i < Table; i += Threads)
        count += keys[i] != Empty;
    int offset, total;
    cub::BlockScan<int, Threads>(scan).ExclusiveSum(count, offset, total);
    if (tid == 0)
        combined_lengths[group] = total;
    for (int i = tid; i < Table; i += Threads) {
        if (keys[i] != Empty) {
            combined[int64_t(group) * output_width + offset] = PackedKeys ? keys[i] >> 4 : keys[i];
            compact_members[offset] = PackedKeys ? keys[i] & 15u : members[i];
            ++offset;
        }
    }
    __syncthreads();
    int lane = tid % 32;
    for (int word = tid / 32; word < output_width / 32; word += Threads / 32) {
        int j = word * 32 + lane;
        unsigned mask = j < total ? compact_members[j] : 0;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            unsigned bits = __ballot_sync(0xffffffff, mask & (1u << r));
            if (lane == 0)
                row_masks[((int64_t(group) * (output_width / 64) + word / 2) * 4 + r) * 2 +
                          word % 2] = bits;
        }
    }
}
} // namespace

template <int Table, bool PackedKeys = false>
void dispatch_prepare(const at::Tensor &indices, const at::Tensor &lengths, int history,
                      at::Tensor &combined, at::Tensor &out_lengths, at::Tensor &masks) {
    constexpr int max_entries = Table >= 4096 ? 2560 : 512;
    constexpr int bytes =
        Table * (PackedKeys ? 4 : 8) + (Table / 2 > max_entries ? Table / 2 : max_entries);
    C10_CUDA_CHECK(cudaFuncSetAttribute(prepare_kernel<Table, PackedKeys>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    prepare_kernel<Table, PackedKeys>
        <<<(indices.size(0) + 3) / 4, Threads, bytes, at::cuda::getCurrentCUDAStream()>>>(
            indices.data_ptr<int>(), lengths.data_ptr<int>(), indices.size(0), indices.size(1),
            history, combined.data_ptr<int>(), out_lengths.data_ptr<int>(),
            reinterpret_cast<unsigned *>(masks.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void prepare_packed_sparse_prefill(const at::Tensor &indices, const at::Tensor &lengths,
                                   int history, at::Tensor &combined, at::Tensor &out_lengths,
                                   at::Tensor &masks) {
    // Four membership bits leave 28 bits for indices. Larger workspaces use
    // separate words, preserving the full int32 index domain.
    if (indices.size(1) == 128)
        dispatch_prepare<1024>(indices, lengths, history, combined, out_lengths, masks);
    else if (history <= (1 << 28))
        dispatch_prepare<8192, true>(indices, lengths, history, combined, out_lengths, masks);
    else
        dispatch_prepare<8192>(indices, lengths, history, combined, out_lengths, masks);
}
