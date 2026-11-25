#pragma once

#include "kernels/params.h"

namespace legacy::smxx::prefill {

// V3.2 FP8 KV pages -> bf16 rows, for the bf16 sparse prefill kernels (see DequantV32Params in kernels/params.h)
void run_dequant_v32_kernel(const DequantV32Params &params);

}
