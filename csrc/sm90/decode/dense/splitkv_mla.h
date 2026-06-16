#pragma once

#include "params.h"
#include "config.h"

namespace sm90 {

template<typename InputT, int HEAD_DIM_K = Config::HEAD_DIM_K>
void run_flash_splitkv_mla_kernel(DenseAttnDecodeParams &params);

}
