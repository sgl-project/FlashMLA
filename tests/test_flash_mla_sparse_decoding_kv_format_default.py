"""
Default KV cache format detection for d_qk = 512 sparse decoding.

When `kv_format` is omitted, flash_mla_with_kvcache detects the format of a
paged fp8 KV cache from its bytes per token: 584 is V4, 288 is V4.1 fp4, and
528 is V4.1 -- the same as upstream FlashMLA. V3.2-no-RoPE is also 528 bytes
per token and must be selected with kv_format="V32_NO_ROPE".

This test runs each cache once with the explicit kv_format and once without it,
and requires bit-identical outputs. The V4.1 case only runs on SM100.
"""

import sys

import torch

import flash_mla

import quant


D_QK = 512
D_V = 512


def build_indices(
    b: int, s_q: int, topk: int, block_table: torch.Tensor, block_size: int, s_kv: int
) -> torch.Tensor:
    """Random abs indices in [0, s_kv) with ~5% invalid (-1), then translated via block_table."""
    abs_indices = torch.empty((b, s_q, topk), dtype=torch.int32, device="cuda")
    for i in range(b):
        for j in range(s_q):
            abs_indices[i, j] = torch.randperm(s_kv, device="cuda")[:topk].to(torch.int32)
    invalid_mask = torch.rand_like(abs_indices.float()) < 0.05
    abs_indices[invalid_mask] = -1

    safe_abs = abs_indices.clone()
    inv = safe_abs == -1
    safe_abs[inv] = 0
    block_within = safe_abs // block_size
    off_within = safe_abs % block_size
    real_block = block_table.gather(1, block_within.view(b, -1).to(torch.int64)).view(b, s_q, topk)
    idx = (real_block * block_size + off_within).to(torch.int32)
    idx[inv] = -1
    return idx


def _bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Byte-for-byte tensor equality (works around NaN != NaN in torch.equal on floats)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    a_bytes = a.contiguous().flatten().view(torch.uint8)
    b_bytes = b.contiguous().flatten().view(torch.uint8)
    return torch.equal(a_bytes, b_bytes)


def run_one(kvcache_layout: quant.KVCacheLayout, kv_format: str, b: int, s_q: int, h_q: int, s_kv: int, topk: int, block_size: int, seed: int) -> bool:
    torch.manual_seed(seed)
    device = "cuda"
    print(f"--- {kv_format} b={b} s_q={s_q} h_q={h_q} s_kv={s_kv} topk={topk} block_size={block_size} seed={seed}")

    assert s_kv % block_size == 0
    blocks_per_seq = s_kv // block_size
    total_blocks = b * blocks_per_seq
    block_table = torch.randperm(total_blocks, dtype=torch.int32, device=device).view(b, blocks_per_seq)

    k = (torch.randn((total_blocks, block_size, 1, D_QK), dtype=torch.bfloat16, device=device) / 10.0).clamp_(-1.0, 1.0)
    q = torch.randn((b, s_q, h_q, D_QK), dtype=torch.bfloat16, device=device).clamp_(-1.0, 1.0)
    kv = quant.quantize_k_cache(k, kvcache_layout)
    indices = build_indices(b, s_q, topk, block_table, block_size, s_kv)
    sm_scale = D_QK ** -0.5

    sched_explicit, _ = flash_mla.get_mla_metadata()
    out_explicit, lse_explicit = flash_mla.flash_mla_with_kvcache(
        q, kv, None, None, D_V,
        sched_explicit, None, sm_scale,
        causal=False, is_fp8_kvcache=True, indices=indices,
        kv_format=kv_format,
    )
    sched_default, _ = flash_mla.get_mla_metadata()
    out_default, lse_default = flash_mla.flash_mla_with_kvcache(
        q, kv, None, None, D_V,
        sched_default, None, sm_scale,
        causal=False, is_fp8_kvcache=True, indices=indices,
    )

    out_ok = _bit_equal(out_explicit, out_default)
    lse_ok = _bit_equal(lse_explicit, lse_default)
    if not out_ok:
        d = (out_explicit.float() - out_default.float()).abs()
        print(f"    out DIFF  max={d.max().item():.4e}  #diff={(d != 0).sum().item()}/{d.numel()}")
    if not lse_ok:
        d = (lse_explicit - lse_default).abs()
        print(f"    lse DIFF  max={d.max().item():.4e}  #diff={(d != 0).sum().item()}/{d.numel()}")

    ok = out_ok and lse_ok
    print(f"    out bit-equal: {out_ok}   lse bit-equal: {lse_ok}   {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    cap = torch.cuda.get_device_capability(device)
    print(f"Compute capability: {cap}")

    layouts = [(quant.KVCacheLayout.V4_FP8Sparse, "V4")]
    if cap[0] >= 10:
        layouts.append((quant.KVCacheLayout.V41_FP8Sparse, "V41"))
    else:
        print("V4.1 KV cache formats need SM100; skipping the V41 case")

    # (b, s_q, h_q, s_kv, topk, block_size, seed)
    S_KV = 128 * 1024
    TOPK = 2048
    cases = [
        (1, 1,  64, S_KV, TOPK, 64, 0),
        (2, 1, 128, S_KV, TOPK, 64, 1),
        (4, 2,  64, S_KV, TOPK, 64, 2),
    ]
    all_ok = True
    for layout, kv_format in layouts:
        for c in cases:
            all_ok &= run_one(layout, kv_format, *c)

    if all_ok:
        print("ALL CASES PASSED (omitted kv_format == explicit kv_format)")
        sys.exit(0)
    else:
        print("SOME CASES FAILED (default kv_format detection differs from the explicit format)")
        sys.exit(1)


if __name__ == "__main__":
    main()
