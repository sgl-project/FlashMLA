#include "common.h"

#include "kernels/sm100/prefill/dense/interface.h"

#ifndef FLASH_MLA_LIBTORCH_ONLY
void register_dense_fwd(pybind11::module_& m) {
    m.def("dense_prefill_fwd",
        &FMHACutlassSM100FwdRun,
        "Run Dense Attention Prefill Forward (cutlass FMHA)");
}
#endif
