"""
Bit-exact equivalence: V32 with K_rope=0 == V32_NO_ROPE

When a V32 KV cache has its RoPE component filled with zeros and Q_rope is
also zero, V32's attention output must be bit-identical to what V32_NO_ROPE
produces from the same NoPE data. Reasoning:
  - The 9th (RoPE) K-tile of V32's QK^T contributes Q_rope * K_rope = 0 * 0 = 0
    in fp32 accumulation, so P is bit-exactly the QK^T of the NoPE dims.
  - V never includes RoPE in either mode, so PV is identical.
  - The tile-wise fp8 quantization is identical on NoPE (same tile size, same
    ue8m0 scale). We verify the first 528 bytes of each V32 token match the
    528 bytes of the V32_NO_ROPE token byte-for-byte.

We force softmax_scale to 512**-0.5 for both runs so the only remaining source
of any diff would be the kernel itself.

Arch coverage note:
  Dispatch is by GPU arch (see csrc/api/sparse_decode.h): sm100f / B200 ->
  Decode_Sm100_Head64[_x2]_Impl, sm90a / H100 -> Decode_Sm90_Impl. These are two
  distinct kernels and the V32-vs-V32_NO_ROPE bit-exactness must hold on both.
  On an H100 only the sm90 path runs; the sm100 path -- including the V32_NO_ROPE
  warp-6 everyone_sync barrier code in csrc/sm100/decode/head64/kernel.cuh -- is
  only exercised on a B200. main() prints the detected compute capability so CI
  logs show which path was actually covered; a green run on H100 alone does NOT
  prove the B200/sm100 path runs. For full coverage, run this test on both H100
  (sm90) and B200 (sm100).
"""

import sys

import torch

import flash_mla

import quant


D_NOPE = 512
D_ROPE = 64
D_V = 512
D_QK_V32 = D_NOPE + D_ROPE          # 576
D_QK_V32_NO_ROPE = D_NOPE           # 512


def build_v32_kv(k_nope_bf16: torch.Tensor) -> torch.Tensor:
    """[num_blocks, block_size, 1, 512] bf16 -> V32 fp8 layout [., ., ., 656]. RoPE zeros."""
    num_blocks, block_size, h_k, _ = k_nope_bf16.shape
    k_full = torch.zeros(
        (num_blocks, block_size, h_k, D_QK_V32),
        dtype=k_nope_bf16.dtype,
        device=k_nope_bf16.device,
    )
    k_full[..., :D_NOPE] = k_nope_bf16
    return quant.quantize_k_cache(k_full, quant.FP8KVCacheLayout.V32_FP8Sparse)


def build_v32_no_rope_kv(k_nope_bf16: torch.Tensor) -> torch.Tensor:
    """[num_blocks, block_size, 1, 512] bf16 -> V32_NO_ROPE fp8 layout [., ., ., 528]."""
    return quant.quantize_k_cache(k_nope_bf16, quant.FP8KVCacheLayout.V32_NO_ROPE_FP8Sparse)


def assert_nope_scale_bytes_equal(kv_v32: torch.Tensor, kv_nr: torch.Tensor):
    """V32 first 528 bytes per token must equal V32_NO_ROPE's 528 bytes exactly."""
    v32_flat = kv_v32.view(kv_v32.shape[0], kv_v32.shape[1], -1)  # [., ., 656]
    nr_flat = kv_nr.view(kv_nr.shape[0], kv_nr.shape[1], -1)      # [., ., 528]
    assert v32_flat.shape[-1] == 656 and nr_flat.shape[-1] == 528
    v32_head = v32_flat[..., :528].contiguous().view(torch.uint8)
    nr_head = nr_flat.contiguous().view(torch.uint8)
    assert torch.equal(v32_head, nr_head), (
        "V32 first-528-byte per-token slice does not match V32_NO_ROPE — "
        "quant paths diverged."
    )


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
    """Byte-for-byte tensor equality (works around NaN != NaN in torch.equal on floats).
    Force a contiguous, stride(-1)==1 buffer before viewing as uint8."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    a_bytes = a.contiguous().flatten().view(torch.uint8)
    b_bytes = b.contiguous().flatten().view(torch.uint8)
    return torch.equal(a_bytes, b_bytes)


def run_one(b: int, s_q: int, h_q: int, s_kv: int, topk: int, block_size: int, seed: int) -> bool:
    torch.manual_seed(seed)
    device = "cuda"
    print(f"--- b={b} s_q={s_q} h_q={h_q} s_kv={s_kv} topk={topk} block_size={block_size} seed={seed}")

    assert s_kv % block_size == 0
    blocks_per_seq = s_kv // block_size
    total_blocks = b * blocks_per_seq
    block_table = torch.randperm(total_blocks, dtype=torch.int32, device=device).view(b, blocks_per_seq)

    # Shared K_nope (bf16), Q_nope (bf16).
    k_nope = (torch.randn((total_blocks, block_size, 1, D_NOPE), dtype=torch.bfloat16, device=device) / 10.0).clamp_(-1.0, 1.0)
    q_nope = torch.randn((b, s_q, h_q, D_NOPE), dtype=torch.bfloat16, device=device).clamp_(-1.0, 1.0)

    # V32 tensors: Q_rope=0, K_rope=0.
    q_v32 = torch.zeros((b, s_q, h_q, D_QK_V32), dtype=torch.bfloat16, device=device)
    q_v32[..., :D_NOPE] = q_nope
    kv_v32 = build_v32_kv(k_nope)          # [., ., ., 656]
    kv_nr = build_v32_no_rope_kv(k_nope)   # [., ., ., 528]
    assert_nope_scale_bytes_equal(kv_v32, kv_nr)

    indices = build_indices(b, s_q, topk, block_table, block_size, s_kv)

    # Force identical softmax_scale (V32 would default to 576**-0.5 which would already
    # break bit-exactness by feeding a different scale into the softmax exp2).
    sm_scale = D_NOPE ** -0.5

    sched_v32, _ = flash_mla.get_mla_metadata()
    out_v32, lse_v32 = flash_mla.flash_mla_with_kvcache(
        q_v32, kv_v32, None, None, D_V,
        sched_v32, None, sm_scale,
        causal=False, is_fp8_kvcache=True, indices=indices,
    )
    sched_nr, _ = flash_mla.get_mla_metadata()
    out_nr, lse_nr = flash_mla.flash_mla_with_kvcache(
        q_nope, kv_nr, None, None, D_V,
        sched_nr, None, sm_scale,
        causal=False, is_fp8_kvcache=True, indices=indices,
    )

    out_ok = _bit_equal(out_v32, out_nr)
    lse_ok = _bit_equal(lse_v32, lse_nr)

    if not out_ok:
        d = (out_v32.float() - out_nr.float()).abs()
        n_diff = (d != 0).sum().item()
        print(f"    out DIFF  max={d.max().item():.4e}  #diff={n_diff}/{d.numel()}")
    if not lse_ok:
        d = (lse_v32 - lse_nr).abs()
        n_diff = (d != 0).sum().item()
        print(f"    lse DIFF  max={d.max().item():.4e}  #diff={n_diff}/{d.numel()}")

    ok = out_ok and lse_ok
    print(f"    out bit-equal: {out_ok}   lse bit-equal: {lse_ok}   {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    cap = torch.cuda.get_device_capability(device)
    arch_name = {9: "SM90/H100", 10: "SM100/B200"}.get(cap[0], f"SM{cap[0]}{cap[1]}")
    print(f"Compute capability: {cap} -> {arch_name} decode path")

    # (b, s_q, h_q, s_kv, topk, block_size, seed)
    S_KV = 128 * 1024
    TOPK = 2048
    cases = [
        (1, 1,  64, S_KV, TOPK, 64, 0),
        (1, 1, 128, S_KV, TOPK, 64, 1),
        (2, 1,  64, S_KV, TOPK, 64, 2),
        (2, 3, 128, S_KV, TOPK, 64, 3),
        (4, 1,  64, S_KV, TOPK, 64, 4),
        (4, 2, 128, S_KV, TOPK, 64, 5),
    ]
    all_ok = True
    for c in cases:
        all_ok &= run_one(*c)

    if all_ok:
        print("ALL CASES PASSED (bit-exact: V32 with K_rope=0 == V32_NO_ROPE)")
        sys.exit(0)
    else:
        print("SOME CASES FAILED (bit-exact equivalence broken)")
        sys.exit(1)


if __name__ == "__main__":
    main()
