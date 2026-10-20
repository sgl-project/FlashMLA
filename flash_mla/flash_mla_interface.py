from typing import Optional, Tuple
import dataclasses

import torch

# This fork builds for CUDA only; upstream's Ascend NPU target is not carried.
from flash_mla._loader import get_ops

@dataclasses.dataclass
class FlashMLASchedMeta:
    """
    A class that stores the tile scheduler metadata of FlashMLA
    """

    @dataclasses.dataclass
    class Config:
        b: int
        s_q: int
        h_q: int
        page_block_size: int
        h_k: int

        causal: bool
        is_fp8_kvcache: bool
        topk: Optional[int]

        extra_page_block_size: Optional[int]
        extra_topk: Optional[int]

        enable_batch_invariant: bool

    have_initialized: bool = False

    config: Optional[Config] = None

    tile_scheduler_metadata: Optional[torch.Tensor] = None   # (num_sm_parts, DecodingSchedMetaSize // 4) == (num_sm_parts, 8), dtype torch.int32.
    num_splits: Optional[torch.Tensor] = None                # (batch_size + 1), dtype torch.int32.


def get_mla_metadata(
    *args,
    **kwargs
) -> Tuple[FlashMLASchedMeta, None]:
    """
    Returns an empty instance of FlashMLASchedMeta. The actual scheduling metadata will be generated during the first invocation of flash_mla_with_kvcache.

    Arguments:
        This function does not need any arguments, but we keep *args and **kwargs to be compatible with the old interface.

    Return:
        A tuple. Due to historical reasons, we return a tuple of (FlashMLASchedMeta, None) now. Only the first element is useful.
    """
    return FlashMLASchedMeta(), None


def get_mla_decoding_metadata(
    seqlens_k: torch.Tensor,
    num_q_tokens_per_head_k: int,
    h_k: int,
    h_q: Optional[int] = None,
    is_fp8_kvcache: bool = False,
    topk: Optional[int] = None,
    use_legacy: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute the split-KV scheduling metadata, hoisted out of the decode call.

    `flash_mla_with_kvcache` fills an empty `FlashMLASchedMeta` itself, which is upstream's
    design; this is for callers that would rather hoist the scheduler out of the decode loop.
    Assign the result onto a `FlashMLASchedMeta` to use it.

    `use_legacy` has to match the decode call it will be paired with: the two schedulers agree
    on every non-empty partition today, but that is a measurement, not a contract either side
    makes. Dense fp8 decoding has a scheduler of its own, selected by `is_fp8_kvcache` without
    a `topk`; it is legacy-only, like the kernel it serves.
    """
    ops = get_ops(use_legacy)
    dense_fp8 = is_fp8_kvcache and topk is None
    if dense_fp8:
        assert use_legacy, "dense fp8 decoding exists only on the legacy side"
        return tuple(ops.get_mla_decoding_metadata_dense_fp8(
            seqlens_k, num_q_tokens_per_head_k, h_k
        ))
    return tuple(ops.get_mla_decoding_metadata(
        seqlens_k, num_q_tokens_per_head_k, h_k, h_q, is_fp8_kvcache, topk
    ))


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: Optional[bool] = None,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    enable_batch_invariant: bool = False,
    use_legacy: bool = False,
    kv_format: Optional[str] = None,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim). bfloat16. `head_dim` must be 512 and
                `num_heads_q` must be 64 or 128.
        k_cache: (num_blocks, page_block_size, num_heads_k, bytes_per_token).
                dtype must be torch.float8_e4m3fn, torch.int8 or torch.uint8, and `num_heads_k`
                must be 1 (only MQA is supported).
                The format is detected from `bytes_per_token`; see the comments below.
                The KV cache must be contiguously valid for sparse attention on sm100. Here "contiguously valid" means that every byte, from the very beginning of the KV cache, till the last byte in the KV cache, is valid memory address to visit (i.e. won't trigger Illegal Memory Access (IMA)). In other words, the KV cache could be a slice of a larger array, but cannot be a list of disjoint memory blocks.
        block_table: currently ignored. We leave it here to be compatible with the old interface
        cache_seqlens: currently ignored. We leave it here to be compatible with the old interface
        head_dim_v: Head_dim of v. Must be 512
        tile_scheduler_metadata: FlashMLASchedMeta, returned by get_mla_metadata. You may reuse the same
                `tile_scheduler_metadata` across different invocations, but only when the tensor shapes and the
                values of topk_length and extra_topk_length remain the same. Note that the values are NOT
                checked at runtime: reusing it with different `topk_length` / `extra_topk_length` values
                silently reuses stale split-KV scheduling metadata.
                `cache_seqlens` is not part of this contract: the decoding path ignores it.
        num_splits: must be None (kept for compatibility with the old interface; the split counts are
                returned inside `tile_scheduler_metadata`).
        softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim_k).
        causal: bool. Must be False, since only sparse attention is supported.
        is_fp8_kvcache: optional bool. None (the default) infers it from `indices`: True for
            sparse decoding, which supports nothing else, and False for dense decoding, which is
            legacy-only and defaults to bf16. Pass True with `indices=None` for dense fp8.
        indices: (batch_size, seq_len_q, topk). KV indices when sparse attention is enabled.
                    Pay attention that indices_in_kvcache[i][j][k] = (the index of the page block where token t resides) * block_size + (the offset of token t among the page block),
                    where t is the k-th token of the j-th q-sequence in the i-th batch.
                    The decoding kernel treats an index as invalid only when it is exactly -1; it performs
                    no upper-bound check, so any other out-of-range positive value produces out-of-bounds
                    (TMA) addresses. Invalid entries must be set to -1.
        attn_sink: Optional[torch.Tensor], (num_heads_q, ), torch.float32. If presented, the final output will be scaled by exp(lse) / (exp(lse) + exp(attn_sink)). Have no affect on the returned softmax_lse. +inf will cause the result to become 0, while -inf has no effect.
        extra_k_cache and extra_indices_in_kvcache: If provided, will attend to these extra tokens in addition to those in k_cache and indices_in_kvcache. Their format requirements are the same as k_cache and indices_in_kvcache respectively.
        topk_length/extra_topk_length: (batch_size, ), torch.int32. If provided, only the leftmost topk_length indices will be processed. Useful when the actual topk for different queries are different so that we can save some computation, compared to masking.
        enable_batch_invariant: bool. If True, the split-KV decoding path is disabled so that results do
                not depend on how the batch is partitioned (`enable_batch_invariant=True` on the first
                invocation must be kept on every reuse of the same `tile_scheduler_metadata`).

    For DeepSeek V4.1:
        head_dim should be 512 while head_dim_v should be 512.
        The format is detected from the last dimension of `k_cache` (i.e. the bytes per token): 528 (V4.1 fp8) or 288 (V4.1 fp4, only valid for an extra cache next to a V4.1 fp8 main cache).
        In both, each token stores its quantized raw data first, followed immediately by its scales:
            - V4.1 (528 Bytes per token): the raw data is 512 float8_e4m3 values, i.e. all 512 dimensions are quantized, including the 64 RoPE ones, so there is no bfloat16 part; the trailing scales are 16 Bytes of float8_e8m0 values, each covering 32 consecutive float8_e4m3 values.
            - V4.1 fp4 (288 Bytes per token): the raw data is 256 Bytes containing 512 e2m1 values (2 values per byte, the even-indexed one in the low nibble); the trailing scales are 32 Bytes of float8_e4m3 values, each covering 16 consecutive e2m1 values.
        See tests/quant.py for quantization and dequantization details.

    Return:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    sched_meta = tile_scheduler_metadata
    indices_in_kvcache = indices
    assert isinstance(sched_meta, FlashMLASchedMeta), "tile_scheduler_metadata must be of type FlashMLASchedMeta"
    assert num_splits is None, "num_splits must be None"
    ops = get_ops(use_legacy)
    if is_fp8_kvcache is None:
        # Upstream defaults this to True, which is right for a build that has only the sparse
        # fp8 kernels. Dense bf16 decoding is back on the legacy side, and it is the one caller
        # that wants False, so the default is inferred from `indices` rather than fixed.
        is_fp8_kvcache = indices_in_kvcache is not None

    if indices_in_kvcache is None:
        # Dense MLA decoding. Upstream deleted it in the 2026.09.30 release, so it exists only on
        # the legacy side, where it is also the reason SM90 is still built.
        assert use_legacy, (
            "Sparse attention is required: `indices` must be provided. Dense MLA decoding was "
            "removed from the current kernels; pass use_legacy=True to reach the legacy one"
        )
        assert kv_format is None, "kv_format does not apply to dense decoding"
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        if is_fp8_kvcache:
            # The fork's own SM90 extension. Its scheduler is a different op, and it wants the
            # buffers up front rather than allocating them, so they are not optional here.
            assert sched_meta.tile_scheduler_metadata is not None, (
                "dense fp8 decoding needs metadata from get_mla_decoding_metadata(..., "
                "is_fp8_kvcache=True, use_legacy=True) before the first call"
            )
            out, lse = ops.fwd_kvcache_mla_fp8(
                q, k_cache, head_dim_v, cache_seqlens, block_table, softmax_scale, causal,
                sched_meta.tile_scheduler_metadata, sched_meta.num_splits,
                descale_q, descale_k,
            )
            return (out, lse)
        assert descale_q is None and descale_k is None, (
            "descale_q/descale_k apply to the dense fp8 path only"
        )
        out, lse, new_meta, new_splits = ops.dense_decode_fwd(
            q, k_cache, head_dim_v, cache_seqlens, block_table, softmax_scale, causal,
            sched_meta.tile_scheduler_metadata, sched_meta.num_splits,
        )
        sched_meta.tile_scheduler_metadata = new_meta
        sched_meta.num_splits = new_splits
        return (out, lse)

    assert not causal, "causal must be False when sparse attention is enabled"
    assert is_fp8_kvcache, "is_fp8_kvcache must be True, since only sparse attention with a quantized KV cache is supported"

    topk = indices_in_kvcache.shape[-1]
    extra_k_page_block_size = extra_k_cache.shape[1] if extra_k_cache is not None else None
    extra_topk = extra_indices_in_kvcache.shape[-1] if extra_indices_in_kvcache is not None else None
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if not sched_meta.have_initialized:
        # Initialize the tile scheduler metadata during the first invocation.
        sched_meta.have_initialized = True
        sched_meta.config = FlashMLASchedMeta.Config(
            b=q.shape[0],
            s_q=q.shape[1],
            h_q=q.shape[2],
            page_block_size=k_cache.shape[1],
            h_k=k_cache.shape[2],

            causal=causal,
            is_fp8_kvcache=is_fp8_kvcache,
            topk=topk,

            extra_page_block_size=extra_k_page_block_size,
            extra_topk=extra_topk,

            enable_batch_invariant=enable_batch_invariant,
        )
    else:
        # Check whether the input arguments are consistent with sched_meta
        helper_msg = " Your input arguments are inconsistent with sched_meta. Please make sure the input arguments are consistent across different invocations of flash_mla_with_kvcache on the same sched_meta."
        assert sched_meta.config is not None
        assert sched_meta.config.b == q.shape[0], "sched_meta.config.b must be equal to batch_size." + helper_msg
        assert sched_meta.config.s_q == q.shape[1], "sched_meta.config.s_q must be equal to seq_len_q." + helper_msg
        assert sched_meta.config.h_q == q.shape[2], "sched_meta.config.h_q must be equal to num_heads_q." + helper_msg
        assert sched_meta.config.page_block_size == k_cache.shape[1], "sched_meta.config.page_block_size must be equal to page_block_size." + helper_msg
        assert sched_meta.config.h_k == k_cache.shape[2], "sched_meta.config.h_k must be equal to num_heads_k." + helper_msg
        assert sched_meta.config.causal == causal, "sched_meta.config.causal must be equal to causal." + helper_msg
        assert sched_meta.config.is_fp8_kvcache == is_fp8_kvcache, "sched_meta.config.is_fp8_kvcache must be equal to is_fp8_kvcache." + helper_msg
        assert sched_meta.config.enable_batch_invariant == enable_batch_invariant, "sched_meta.config.enable_batch_invariant must be equal to enable_batch_invariant." + helper_msg
        assert sched_meta.config.topk == topk, "sched_meta.config.topk must be equal to the last dim of indices_in_kvcache." + helper_msg
        assert sched_meta.config.extra_page_block_size == extra_k_page_block_size, "sched_meta.config.extra_page_block_size must be equal to the page_block_size of extra_k_cache." + helper_msg
        assert sched_meta.config.extra_topk == extra_topk, "sched_meta.config.extra_topk must be equal to the last dim of extra_indices_in_kvcache." + helper_msg

    if use_legacy:
        assert not enable_batch_invariant, (
            "enable_batch_invariant is a 2026.09.30 feature and only exists on the current kernels"
        )
        out, lse, new_tile_scheduler_metadata, new_num_splits = ops.sparse_decode_fwd(
            q, k_cache, indices_in_kvcache, topk_length, attn_sink,
            sched_meta.tile_scheduler_metadata, sched_meta.num_splits,
            extra_k_cache, extra_indices_in_kvcache, extra_topk_length,
            head_dim_v, softmax_scale, kv_format
        )
    else:
        assert kv_format is None, (
            "kv_format only applies to the legacy kernels: the current ones accept the V4.1 "
            "formats only, so there is nothing to disambiguate"
        )
        out, lse, new_tile_scheduler_metadata, new_num_splits = ops.sparse_decode_fwd(
            q, k_cache, indices_in_kvcache, topk_length, attn_sink,
            sched_meta.tile_scheduler_metadata, sched_meta.num_splits,
            extra_k_cache, extra_indices_in_kvcache, extra_topk_length,
            head_dim_v, softmax_scale, enable_batch_invariant
        )
    sched_meta.tile_scheduler_metadata = new_tile_scheduler_metadata
    sched_meta.num_splits = new_num_splits
    return (out, lse)


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    use_legacy: bool = False,
    return_log2_lse: Optional[bool] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16. `d_qk` must be 512 and `h_q` must be 64 or 128.
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512
        attn_sink: optional, [h_q], float32.
            If attn_sink is provided, when computing output, output will be additionally multiplied by exp(lse) / (exp(lse) + exp(attn_sink)).
            +-inf in attn_sink will be handled normally (i.e., -inf has no effect, +inf will make corresponding output all zeros).
            This argument has no effect on lse and max_logits.
        topk_length: optional, [s_q], int32. If provided, the i-th q token will only attend to k tokens specified by indices[i, :, :topk_length[i]], ignoring later k/v tokens (even if provided in indices).
            In extremely rare cases (topk_length provided, there is a valid topk index between topk_length[i] ~ s_kv, and that topk index points to a k token containing NaN), operator output will contain NaN, so please avoid this situation.
        return_log2_lse: optional. None (the default) returns both statistics in natural-log
            units; False returns None for both; True has the kernel emit them in base 2 and is
            legacy-only. The output is the same either way.
            False is a saving only on the legacy side, where the kernel then skips allocating and
            writing the statistics; the current kernels always write them, so there it only hides
            them to keep the signature uniform.

    Returns:
        (output, max_logits, lse)
        Please refer to tests/ref.py for the precise definitions of these parameters.
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float, or None when return_log2_lse is False
        - lse: [s_q, h_q], float, log-sum-exp of attention scores, or None when
            return_log2_lse is False
    """
    ops = get_ops(use_legacy)
    if not use_legacy:
        assert return_log2_lse is not True, (
            "return_log2_lse=True is a legacy-kernel feature: the current kernels emit both "
            "statistics in natural-log units, and converting them here would cost the launches "
            "the kernel avoids"
        )
        out, max_logits, lse = ops.sparse_prefill_fwd(
            q, kv, indices, sm_scale, d_v, attn_sink, topk_length
        )
        if return_log2_lse is False:
            return (out, None, None)
        return (out, max_logits, lse)
    return ops.sparse_prefill_fwd(
        q, kv, indices, sm_scale, d_v, attn_sink, topk_length,
        return_log2_lse is True, return_log2_lse is not False,
    )


def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ops = get_ops(use_legacy=False)
    qo_total_len, num_qo_heads, head_dim_qk = q.shape
    head_dim_vo = v.shape[-1]


    mask_mode_code = 1 if causal else 0
    if softmax_scale is None:
        softmax_scale = head_dim_qk ** (-0.5)

    if out is None:
        out = torch.empty(qo_total_len, num_qo_heads, head_dim_vo, device=q.device, dtype=q.dtype)
    if lse is None:
        # Make lse contiguous on seqlen dim
        lse = torch.empty(num_qo_heads, qo_total_len, device=q.device, dtype=torch.float32).T

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    ops.dense_prefill_fwd(
        workspace_buffer,
        q,
        k,
        v,
        cu_seqlens_qo,
        cu_seqlens_kv,
        out,
        lse,
        mask_mode_code,
        softmax_scale,
        max_seqlen_qo,
        max_seqlen_kv,
        is_varlen,
    )

    return out, lse


def _flash_attn_varlen_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ops = get_ops(use_legacy=False)
    qo_total_len, num_qo_heads, head_dim_qk = q.shape
    kv_total_len, num_kv_heads, head_dim_vo = v.shape


    # TODO: fix bwd GQA
    if num_qo_heads != num_kv_heads:
        raise ValueError(f"SM100 bwd doesn't support GQA now. num_qo_heads: {num_qo_heads}, num_kv_heads: {num_kv_heads}.")

    mask_mode_code = 1 if causal else 0
    if softmax_scale is None:
        softmax_scale = head_dim_qk ** (-0.5)

    if dq is None:
        dq = torch.empty(qo_total_len, num_qo_heads, head_dim_qk, device=q.device, dtype=q.dtype)
    if dk is None:
        dk = torch.empty(kv_total_len, num_kv_heads, head_dim_qk, device=q.device, dtype=q.dtype)
    if dv is None:
        dv = torch.empty(kv_total_len, num_kv_heads, head_dim_vo, device=q.device, dtype=q.dtype)

    # The C++ side takes the q length from the problem shape: `max_seqlen_qo` for variable-length
    # batches, but `total_seqlen_q / batch_size` for fixed-length ones
    # (csrc/cuda_kernels/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cuh), so the workspace must be
    # sized with that same value.
    bs = cu_seqlens_qo.shape[0] - 1
    seqlen_qo_for_workspace = max_seqlen_qo if is_varlen else qo_total_len // bs
    max_seqlen_qo_aligned = (seqlen_qo_for_workspace + 7) // 8 * 8
    workspace_bytes = 0
    workspace_bytes += 4 * bs * max_seqlen_qo_aligned * num_qo_heads * head_dim_qk  # dQ_acc
    workspace_bytes += 4 * max_seqlen_qo_aligned * bs * num_qo_heads * 2  # sum_OdO and scaled_lse
    workspace_buffer = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
    ops.dense_prefill_bwd(
        workspace_buffer,
        do,
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_qo,
        cu_seqlens_kv,
        dq,
        dk,
        dv,
        mask_mode_code,
        softmax_scale,
        max_seqlen_qo,
        max_seqlen_kv,
        is_varlen,
    )

    return dq, dk, dv


class FlashAttnVarlenFunc(torch.autograd.Function):
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_qo: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        max_seqlen_qo: int,
        max_seqlen_kv: int,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        is_varlen: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, lse = _flash_attn_varlen_forward(
            q, k, v,
            cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
            causal=causal, softmax_scale=softmax_scale,
            is_varlen=is_varlen,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_qo, cu_seqlens_kv)
        ctx.max_seqlen_qo = max_seqlen_qo
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.is_varlen = is_varlen
        return out, lse

    def backward(
        ctx,
        do: torch.Tensor,
        dlse: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        del dlse  # LSE doesn't support backward currently
        q, k, v, out, lse, cu_seqlens_qo, cu_seqlens_kv = ctx.saved_tensors
        dq, dk, dv = _flash_attn_varlen_backward(
            do, q, k, v, out, lse,
            cu_seqlens_qo, cu_seqlens_kv, ctx.max_seqlen_qo, ctx.max_seqlen_kv,
            causal=ctx.causal, softmax_scale=ctx.softmax_scale,
            is_varlen=ctx.is_varlen,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    deterministic: bool = False,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dense (MHA) attention forward with variable-length batching, on sm100/sm103 only.

    Args:
        q: [total_qo_tokens, num_qo_heads, head_dim_qk], bfloat16. `head_dim_qk` must be 192 or 128.
        k: [total_kv_tokens, num_kv_heads, head_dim_qk], bfloat16.
        v: [total_kv_tokens, num_kv_heads, head_dim_vo], bfloat16. `head_dim_vo` must be 128.
            Only the (head_dim_qk, head_dim_vo) pairs (192, 128) and (128, 128) are instantiated; every
            other pair would make the kernel return without writing the outputs.
        cu_seqlens_qo: [batch_size + 1], int32, cumulative query sequence lengths, starting at 0 and
            ending at total_qo_tokens.
        cu_seqlens_kv: [batch_size + 1], int32, cumulative key/value sequence lengths.
        max_seqlen_qo: int. Maximum query sequence length in the batch.
        max_seqlen_kv: int. Maximum key/value sequence length in the batch.
        dropout_p: must be 0.0 (dropout is not implemented).
        softmax_scale: optional float. Defaults to head_dim_qk ** (-0.5).
        causal: bool. Whether to apply a causal attention mask.
        deterministic: must be False. The deterministic backward mode is not implemented, and the
            backward pass does not guarantee a bitwise-reproducible dq.
        is_varlen: bool. If True the batch is variable-length and `cu_seqlens_*`/`max_seqlen_*` are
            used; if False the sequences are treated as being of equal length
            (`total_tokens / batch_size`, which is what the backward pass uses internally).

    Returns:
        (out, lse)
        - out: [total_qo_tokens, num_qo_heads, head_dim_vo], bfloat16
        - lse: [total_qo_tokens, num_qo_heads], float32, natural log (base e), contiguous on the
          sequence-length dimension (stride(0) == 1)

    Note:
        The backward pass supports only `num_qo_heads == num_kv_heads` (no GQA), and it requires the
        same dtypes and head dims as the forward pass.
    """
    assert dropout_p == 0.0, "dropout is not supported, `dropout_p` must be 0.0"
    assert not deterministic, "the `deterministic` flag is not supported: the deterministic backward mode is not implemented and dq is not guaranteed to be bitwise reproducible"
    return FlashAttnVarlenFunc.apply(
        q, k, v,
        cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
        causal, softmax_scale, is_varlen,
    )
