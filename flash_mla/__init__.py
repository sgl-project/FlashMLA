import os as _os

with open(_os.path.join(_os.path.dirname(__file__), "VERSION")) as _f:
    __version__ = _f.read().strip()

from flash_mla._loader import has_legacy
from flash_mla.flash_mla_interface import (
    get_mla_metadata,
    get_mla_decoding_metadata,
    flash_mla_with_kvcache,
    flash_attn_varlen_func,
    flash_mla_sparse_fwd
)

from . import fused_norm_rope_attn_rope_cast

__all__ = [
    "get_mla_metadata",
    "get_mla_decoding_metadata",
    "flash_mla_with_kvcache",
    "flash_attn_varlen_func",
    "flash_mla_sparse_fwd",
    "fused_norm_rope_attn_rope_cast",
    "has_legacy",
]
