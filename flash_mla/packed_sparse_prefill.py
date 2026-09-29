from typing import Optional, Tuple

import torch

from . import cuda


def flash_mla_packed_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    sm_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BF16 sparse prefill for 16 real heads on 148-SM B200 and B300 GPUs.

    q is [tokens, 16, 512], including strided views of padded Q; kv is
    [history, 1, 512]. indices is contiguous int32 [tokens, 128 or 640],
    and topk_length is contiguous int32 [tokens]. Out-of-range indices and
    entries past each row's length are ignored; duplicate indices retain
    their multiplicity. The optional sink is float32 [16].

    Returns output [tokens, 16, 512], max_logits [tokens, 16], and lse
    [tokens, 16]. Statistics use natural logs and exclude the attention sink,
    matching the native sparse-prefill API. Empty rows return zero output,
    max_logits=-inf and lse=+inf. This explicit entry has no size gate;
    callers choose whether packing is beneficial for their workload.
    """
    return tuple(
        cuda.packed_sparse_prefill_fwd(q, kv, indices, topk_length, sm_scale, attn_sink)
    )
