#pragma once

#include "kernels/params.h"

namespace legacy::smxx::prefill {

void run_dequant_v32_kernel(const DequantV32Params &params);

}
