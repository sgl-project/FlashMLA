#pragma once

#include "kernels/params.h"
#include "config.h"

namespace legacy::sm90::decode::dense {

template<typename InputT, int HEAD_DIM_K = Config::HEAD_DIM_K>
void run_flash_splitkv_mla_kernel(DenseAttnDecodeParams &params);

}
