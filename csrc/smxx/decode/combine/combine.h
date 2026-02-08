#pragma once

#include "params.h"
#include "flashmla_utils.h"

namespace smxx::decode {

template<typename ElementT>
void run_flash_mla_combine_kernel(CombineParams &params);

}
