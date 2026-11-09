#pragma once

#include <algorithm>
#include <cstdlib>

// Split-KV partitions (CTAs per head block) of the SM100 head64 sparse decode kernel, shared by its tile scheduler
// metadata (get_mla_decoding_metadata, FlashMLASchedMeta) and the kernel launch so that they always agree.
//
// With more than one block of 64 heads, the SMs are divided among the head blocks while the requests alone cannot
// fill the divided partitions (b < num_sms / s_q / head_blocks); this avoids a second wave of CTAs. With more
// requests the greedy schedule over fewer partitions can split requests needlessly, so the undivided schedule (the
// one the old two-launch path used) is kept. FLASHMLA_DIVIDE_PARTS=0/1 forces either.
inline int sm100_sparse_decode_num_sm_parts(int num_sms, int b, int s_q, int h_q) {
    const int head_blocks = (h_q + 63) / 64;
    const int parts = std::max(num_sms / s_q, 1);
    const int divided = std::max(parts / head_blocks, 1);
    const char* env = std::getenv("FLASHMLA_DIVIDE_PARTS");
    const bool divide = env != nullptr ? env[0] == '1' : (head_blocks > 1 && b < divided);
    return divide ? divided : parts;
}
