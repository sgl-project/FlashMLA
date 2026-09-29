"""Standalone regression for GLM-5.2's 416-byte sparse-decode cache ABI."""

import pytest
import torch

from flash_mla import flash_mla_with_kvcache_nvfp4
from quant import _dequantize_e2m1


@pytest.mark.parametrize("batch", [1, 2])
def test_glm52_nvfp4_sparse_decode(batch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100/SM103")
    torch.manual_seed(35237)
    device = "cuda"
    pages, page_size, topk = 4, 64, 2048
    packed = torch.randint(0, 256, (pages, page_size, 256), device=device, dtype=torch.uint8)
    sf = torch.rand((pages, page_size, 32), device=device).mul_(0.5).add_(0.5).to(torch.float8_e4m3fn)
    rope = torch.randn((pages, page_size, 64), device=device, dtype=torch.bfloat16).mul_(0.125)
    cache = torch.empty((pages, page_size, 1, 416), device=device, dtype=torch.uint8)
    cache[..., :256] = packed.unsqueeze(2)
    cache[..., 256:288] = sf.view(torch.uint8).unsqueeze(2)
    cache[..., 288:] = rope.view(torch.uint8).reshape(pages, page_size, 1, 128)
    scale = torch.tensor([100.0 / (448 * 6)], device=device)
    q = torch.randn((batch, 1, 64, 576), device=device, dtype=torch.bfloat16).mul_(0.125)
    indices = torch.randint(0, pages * page_size, (batch, 1, topk), device=device, dtype=torch.int32)
    indices[..., 0] = -1
    indices[..., 1] = pages * page_size + 17
    lengths = torch.full((batch,), topk, device=device, dtype=torch.int32)
    if batch > 1:
        lengths[-1] = 1792
    sink = torch.full((64,), 7.0, device=device) if batch > 1 else None
    sm_scale = 1 / (576**0.5)

    out, lse, metadata, splits = flash_mla_with_kvcache_nvfp4(
        q, cache, scale, indices, topk_length=lengths, attn_sink=sink, softmax_scale=sm_scale
    )
    assert out.shape == (batch, 1, 64, 512)
    assert lse.shape == (batch, 64, 1)
    assert metadata is not None and splits is not None
    assert metadata.size(0) <= torch.cuda.get_device_properties(device).multi_processor_count

    flat_indices = indices[:, 0].long()
    valid = ((flat_indices >= 0) & (flat_indices < pages * page_size) &
             (torch.arange(topk, device=device)[None, :] < lengths[:, None]))
    safe = flat_indices.masked_fill(~valid, 0)
    row = cache.reshape(-1, 416)[safe]
    codes = row[..., :256]
    codes = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(batch, topk, 512)
    sf_row = row[..., 256:288].view(torch.float8_e4m3fn).float()
    effective_sf = (sf_row.half() * scale.half()).float().repeat_interleave(16, dim=-1)
    latent = (_dequantize_e2m1(codes) * effective_sf).to(torch.bfloat16).float()
    rope_row = row[..., 288:].contiguous().view(torch.bfloat16).float()
    kv = torch.cat((latent, rope_row), dim=-1)
    logits = torch.einsum("bhd,bkd->bhk", q[:, 0].float(), kv) * sm_scale
    logits.masked_fill_(~valid[:, None, :], -torch.inf)
    probs = logits.softmax(dim=-1)
    expected = torch.einsum("bhk,bkd->bhd", probs, latent)
    if sink is not None:
        mass = torch.exp(torch.logsumexp(logits, dim=-1))
        expected *= (mass / (mass + sink.exp()[None, :])).unsqueeze(-1)
    expected = expected.to(torch.bfloat16)
    # The kernel rounds per-block probabilities to BF16 and uses split-KV reduction.
    torch.testing.assert_close(out[:, 0], expected, atol=0.035, rtol=0.035)
    assert torch.isfinite(lse).all()

    # Reuse caller-owned metadata, including across CUDA graph replays.
    graph_out, graph_lse, _, _ = flash_mla_with_kvcache_nvfp4(
        q, cache, scale, indices, topk_length=lengths, attn_sink=sink,
        tile_scheduler_metadata=metadata, num_splits=splits, softmax_scale=sm_scale,
    )
    torch.testing.assert_close(graph_out, out, atol=0.01, rtol=0.01)
    torch.testing.assert_close(graph_lse, lse, atol=0.01, rtol=0.01)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_out, replay_lse, _, _ = flash_mla_with_kvcache_nvfp4(
            q, cache, scale, indices, topk_length=lengths, attn_sink=sink,
            tile_scheduler_metadata=metadata, num_splits=splits, softmax_scale=sm_scale,
        )
    graph.replay()
    torch.testing.assert_close(replay_out, out, atol=0.01, rtol=0.01)
    torch.testing.assert_close(replay_lse, lse, atol=0.01, rtol=0.01)


def test_v32_fp8_sparse_decode_unchanged():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100/SM103")
    from flash_mla.cuda import sparse_decode_fwd

    q = torch.randn(1, 1, 64, 576, device="cuda", dtype=torch.bfloat16).mul_(0.125)
    cache = torch.empty((4, 64, 1, 656), device="cuda", dtype=torch.uint8)
    fp8 = torch.randn((4, 64, 512), device="cuda").mul_(0.125).to(torch.float8_e4m3fn)
    cache[..., :512] = fp8.view(torch.uint8).unsqueeze(2)
    cache[..., 512:528] = torch.ones((4, 64, 1, 4), device="cuda").view(torch.uint8).reshape(4, 64, 1, 16)
    rope = torch.randn((4, 64, 64), device="cuda", dtype=torch.bfloat16).mul_(0.125)
    cache[..., 528:] = rope.view(torch.uint8).reshape(4, 64, 1, 128)
    indices = torch.randint(0, 256, (1, 1, 2048), device="cuda", dtype=torch.int32)
    out, lse, _, _ = sparse_decode_fwd(q, cache, indices, None, None, None, None,
                                        None, None, None, 512, 1 / 576**0.5, None)
    assert out.shape == (1, 1, 64, 512)
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()


def test_v41_fp4_extra_cache_unchanged():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100/SM103")
    from flash_mla.cuda import sparse_decode_fwd

    q = torch.randn(1, 1, 64, 512, device="cuda", dtype=torch.bfloat16).mul_(0.125)
    # Unlike GLM-5.2, V4.1 stores all packed data rows first, then all scale rows.
    primary = torch.zeros((2, 64, 1, 528), device="cuda", dtype=torch.uint8)
    extra = torch.zeros((2, 64, 1, 288), device="cuda", dtype=torch.uint8)
    indices = torch.randint(0, 128, (1, 1, 64), device="cuda", dtype=torch.int32)
    extra_indices = torch.randint(0, 128, (1, 1, 64), device="cuda", dtype=torch.int32)
    out, lse, _, _ = sparse_decode_fwd(q, primary, indices, None, None, None, None,
                                       extra, extra_indices, None, 512, 1 / 512**0.5, "V41")
    assert out.shape == (1, 1, 64, 512)
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()
