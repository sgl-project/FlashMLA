#pragma once

#include "params.h"

// Layout of paged quantized KV cache. A page block holds page_block_size tokens as two byte arrays, one row per token:
//   [page_block_size, TMA_K_STRIDE]                        token data: D_FP4 / 2 + D_FP8 bytes of quantized values,
//                                                          then the D_BF16 bf16 (RoPE) values of V3.2 / V4
//   [page_block_size, NUM_SCALES_EACH_TOKEN * SCALE_BYTES] scales
// Exception: the V3.2 formats have no scale array; their 4 fp32 scales sit inside the token data, right after the NoPE part
// (V3.2 then has its RoPE part; V3.2-no-RoPE ends there).
// Reference quantizer / dequantizer: tests/quant.py
template<ModelType MT>
struct KVCacheFormat {
    static constexpr ModelType MODEL_TYPE = MT;
    static constexpr bool IS_V32 = MT == ModelType::V32 || MT == ModelType::V32_NO_ROPE;   // fp32 scales inside the token data
    static constexpr bool IS_FP4 = MT == ModelType::V41_FP4;
    static constexpr int D_QK = MT == ModelType::V32 ? 576 : 512;
    static constexpr int D_ROPE = MT == ModelType::V32_NO_ROPE ? 0 : 64;
    static constexpr int D_NOPE = D_QK - D_ROPE;
    static constexpr int D_FP4 = IS_FP4 ? D_QK : 0;                                     // Dimensions stored as fp4 e2m1
    static constexpr int D_FP8 = IS_FP4 ? 0 : (MT == ModelType::V41 ? D_QK : D_NOPE);   // Dimensions stored as fp8 e4m3, needing dequant (V3.2-no-RoPE: D_NOPE == D_QK)
    static constexpr int D_BF16 = D_QK - D_FP4 - D_FP8;                                 // Dimensions stored as bf16 (the RoPE part of V3.2 / V4), not needing dequant
    static constexpr int QUANT_TILE_SIZE = IS_FP4 ? 16 : (MT == ModelType::V41 ? 32 : (MT == ModelType::V4 ? 64 : 128));   // Dimensions sharing one scale
    static constexpr int NUM_SCALES_EACH_TOKEN = (IS_V32 ? D_NOPE : D_QK) / QUANT_TILE_SIZE;   // 4 / 4 / 8 (7 + 1 byte of padding) / 16 / 32
    static constexpr int SCALE_BYTES = IS_V32 ? 4 : 1;                                  // fp32 (V3.2), ue8m0 (V4 / V4.1) or e4m3 (fp4)
    static constexpr int SCALE_SMEM_BYTES = IS_V32 ? 2 : 1;                             // Bytes per scale in shared memory: bf16 (V3.2), ue8m0 or e4m3
    static constexpr int QUANT_BYTES = D_FP4 / 2 + D_FP8;                               // The quantized (fp4 / fp8) data of a token
    // Bytes between two tokens in the data region of a page block: 656 / 528 / 576 / 512 / 256. The stride of the tensor maps of
    // the quantized part, so it must be >= 256 for the int32 TMA coordinates to cover a whole KV cache
    static constexpr int TMA_K_STRIDE = QUANT_BYTES + (IS_V32 ? NUM_SCALES_EACH_TOKEN * SCALE_BYTES : 0) + 2 * D_BF16;
    // 656 (V3.2) / 528 (V3.2-no-RoPE) / 584 (V4) / 528 (V4.1) / 288 (V4.1 fp4). NOTE V3.2-no-RoPE and V4.1 collide, so
    // detect_kv_cache_format_for_headdim_512 cannot tell them apart -- see the kv_format argument of sparse_decode_fwd
    static constexpr int BYTES_PER_TOKEN = TMA_K_STRIDE + (IS_V32 ? 0 : NUM_SCALES_EACH_TOKEN * SCALE_BYTES);
};

// Runtime counterpart of KVCacheFormat<MT>::BYTES_PER_TOKEN
constexpr int kv_cache_bytes_per_token(ModelType mt) {
    switch (mt) {
        case ModelType::V32: return KVCacheFormat<ModelType::V32>::BYTES_PER_TOKEN;
        case ModelType::V4: return KVCacheFormat<ModelType::V4>::BYTES_PER_TOKEN;
        case ModelType::V41: return KVCacheFormat<ModelType::V41>::BYTES_PER_TOKEN;
        case ModelType::V41_FP4: return KVCacheFormat<ModelType::V41_FP4>::BYTES_PER_TOKEN;
        case ModelType::V32_NO_ROPE: return KVCacheFormat<ModelType::V32_NO_ROPE>::BYTES_PER_TOKEN;
    }
    return 0;
}

// The (kv, extra_kv) format pairs that exist: extra_kv has the format of kv, or is the V4.1 fp4 cache next to a V4.1 (fp8) kv.
constexpr bool is_valid_kv_format_pair(ModelType kv, ModelType extra_kv) {
    return extra_kv == kv || (kv == ModelType::V41 && extra_kv == ModelType::V41_FP4);
}

template<ModelType KV, ModelType EXTRA_KV = KV>
struct KVFormatPair {
    static_assert(is_valid_kv_format_pair(KV, EXTRA_KV));
    static constexpr ModelType kv = KV, extra_kv = EXTRA_KV;
};

// A list of KVFormatPair, see dispatch_kv_formats in csrc/api/common.h
template<typename... Pairs>
struct KVFormatPairs {};
