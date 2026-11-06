"""prefill(out_sf_flashinfer_layout=True) writes the same FP8 output and scale bytes as the default, in FlashInfer's 128x4 layout."""
import pytest
import torch

import flash_mla
from flash_mla import fused_norm_rope_attn_rope_cast as fused


def _is_sm100f():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def _run(s_q, flashinfer_layout, h_q=64, d_qk=512, topk=512, s_kv=4096, n_wv_group=8):
    g = torch.Generator(device="cuda").manual_seed(s_q)
    q = torch.randn(s_q, h_q, d_qk, device="cuda", dtype=torch.bfloat16, generator=g)
    kv = torch.randn(s_kv, 1, d_qk, device="cuda", dtype=torch.bfloat16, generator=g)
    indices = torch.randint(0, s_kv, (s_q, 1, topk), device="cuda", dtype=torch.int32, generator=g)
    positions = torch.randint(0, 4096, (s_q,), device="cuda", dtype=torch.int32, generator=g)
    cos_sin = torch.randn(4096, 64, device="cuda", generator=g)
    sink = torch.randn(h_q, device="cuda", generator=g)
    return fused.prefill(
        True, 1e-6, positions, False, 64, cos_sin, n_wv_group, 32, True, True, True,
        q, kv, indices, d_qk ** -0.5, 512, attn_sink=sink, out_sf_flashinfer_layout=flashinfer_layout,
    )


def _swizzle_128x4(sf_words):
    """Default out_sf [s_q, n_group, words] -> FlashInfer 128x4 words per group, flat."""
    s_q, n_group, words = sf_words.shape
    w = sf_words.permute(1, 0, 2)  # [group, row, word]
    w = w.reshape(n_group, s_q // 128, 4, 32, words)  # row = (block, r/32 % 4, r % 32)
    return w.permute(0, 1, 4, 3, 2).contiguous().view(-1)  # word = block*4096 + w*128 + (r%32)*4 + (r/32)%4


@pytest.mark.skipif(not _is_sm100f(), reason="needs an SM100-family GPU")
@pytest.mark.parametrize("s_q", [128, 512, 1024, 4096])
def test_out_sf_flashinfer_layout(s_q):
    out_fp8, out_sf, max_logits, lse = _run(s_q, False)
    fi_fp8, fi_sf, fi_max_logits, fi_lse = _run(s_q, True)
    assert torch.equal(out_fp8.view(torch.uint8), fi_fp8.view(torch.uint8))
    assert torch.equal(lse, fi_lse) and torch.equal(max_logits, fi_max_logits)
    raw = torch.as_strided(fi_sf, (fi_sf.numel(),), (1,))
    assert torch.equal(raw, _swizzle_128x4(out_sf))


@pytest.mark.skipif(not _is_sm100f(), reason="needs an SM100-family GPU")
def test_out_sf_flashinfer_layout_rejects_ragged_rows():
    with pytest.raises(RuntimeError, match="multiple of 128"):
        _run(130, True)
