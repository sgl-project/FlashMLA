"""Load the extensions and expose their op namespaces.

Neither extension defines `PyInit_*`: both register through `STABLE_TORCH_LIBRARY`, which runs
during static initialisation and needs no CPython API at all -- that is what lets a single wheel
serve every CPython >= 3.10. So they are loaded with `torch.ops.load_library` (a dlopen) rather
than imported.

Two extensions, two op namespaces:

  torch.ops.flash_mla          upstream's current kernels: V4.1 on SM100
  torch.ops.flash_mla_legacy   the kernels upstream dropped in its 2026.09.30 release:
                               SM90, and the V3.2 / V3.2-no-RoPE / V4 KV cache formats

The legacy half is optional, so a build made with FLASH_MLA_BUILD_LEGACY=0 still imports; asking
for a legacy kernel then raises instead of failing at load time.
"""

import glob
import os
from typing import Optional

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

_LEGACY_UNAVAILABLE: Optional[str] = None


def _load(stem: str) -> bool:
    # The file name carries the ABI tag, which differs between an abi3 build and a
    # CPython-specific one, so it is matched rather than spelled out.
    matches = sorted(glob.glob(os.path.join(_HERE, f"{stem}.*.so")))
    if not matches:
        return False
    torch.ops.load_library(matches[0])
    return True


if not _load("cuda"):
    raise ImportError(
        f"FlashMLA's CUDA extension was not found in {_HERE}. "
        "Build it with `pip install -v . --no-build-isolation`."
    )

try:
    _has_legacy = _load("_C_legacy")
except OSError as exc:  # a library that is present but will not load is worth reporting
    _has_legacy = False
    _LEGACY_UNAVAILABLE = str(exc)
else:
    if not _has_legacy:
        _LEGACY_UNAVAILABLE = "this wheel was built with FLASH_MLA_BUILD_LEGACY=0"


def get_ops(use_legacy: bool):
    """The op namespace to call, picked by the caller's `use_legacy`."""
    if not use_legacy:
        return torch.ops.flash_mla
    if not _has_legacy:
        raise RuntimeError(
            "The legacy FlashMLA kernels (SM90, and the V3.2 / V3.2-no-RoPE / V4 KV cache "
            f"formats) are not available: {_LEGACY_UNAVAILABLE}."
        )
    return torch.ops.flash_mla_legacy


def has_legacy() -> bool:
    return _has_legacy
