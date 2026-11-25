#include "dequant_v32.h"

#include <cstdio>
#include <cstdlib>
#include <utility>

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

#include "kernels/utils.h"

namespace legacy::smxx::prefill {

namespace {

static constexpr int NUM_THREADS = 256;
static constexpr int ROWS_PER_CTA = NUM_THREADS / 32;
static constexpr int MAX_CTAS = 4096;
static constexpr int BYTES_PER_TOKEN = 656;
static constexpr int TOKENS_PER_PAGE = 64;

__global__ void __launch_bounds__(NUM_THREADS)
dequant_v32_kernel(const DequantV32Params params) {
    const int lane = threadIdx.x & 31;
    const int64_t warps = (int64_t)gridDim.x * ROWS_PER_CTA;
    for (int64_t r = (int64_t)blockIdx.x * ROWS_PER_CTA + (threadIdx.x >> 5); r < params.n; r += warps) {
        const int64_t slot = params.rows ? (int64_t)params.rows[r] : r;
        const uint8_t* src = params.kv + (slot >> 6) * params.stride_page + (slot & 63) * BYTES_PER_TOKEN;
        uint16_t* dst = params.out + r * 576;
        // This lane's 16 NoPE bytes belong to quantization tile lane / 8
        const uint4 raw = *(const uint4*)(src + 16 * lane);
        const float scale = *(const float*)(src + 512 + 4 * (lane >> 3));
        const uint8_t* b = (const uint8_t*)&raw;
        uint4 o[2];
        uint16_t* ob = (uint16_t*)o;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            __nv_fp8_e4m3 f;
            f.__x = b[i];
            ob[i] = __bfloat16_as_ushort(__float2bfloat16(float(f) * scale));
        }
        *(uint4*)(dst + 16 * lane) = o[0];
        *(uint4*)(dst + 16 * lane + 8) = o[1];
        if (lane < 8) *(uint4*)(dst + 512 + 8 * lane) = *(const uint4*)(src + 528 + 16 * lane);
    }
}

}  // namespace

void run_dequant_v32_kernel(const DequantV32Params &params) {
    static_assert(TOKENS_PER_PAGE == 64);   // The kernel splits a slot id with >> 6 and & 63
    int64_t num_ctas = (params.n + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
    if (num_ctas > MAX_CTAS) num_ctas = MAX_CTAS;
    dequant_v32_kernel<<<(unsigned)num_ctas, NUM_THREADS, 0, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}
