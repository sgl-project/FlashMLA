"""Native defaults stay natural-log; explicit modes fuse base-2 or omit stats."""

import math

import flash_mla.cuda as flash_mla_cuda
import pytest
import torch

from flash_mla import flash_mla_sparse_fwd

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("dim", [512, 576])
@pytest.mark.parametrize("topk", [128, 512, 2048])
@pytest.mark.parametrize("with_sink", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("return_log2_lse", [None, True, False])
def test_native_prefill_stats_units(
    heads, dim, topk, with_sink, masked, return_log2_lse
):
    if torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("Sparse prefill requires SM90 or SM100-family")
    q = torch.ones(3, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.full((1024, 1, dim), 0.125, device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros(3, 1, topk, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([0, 65, topk], device="cuda", dtype=torch.int32)
    if masked:
        indices[:, :, ::4] = -1
        indices[:, :, 1::4] = kv.shape[0]
        indices[1].fill_(-1)  # Empty attention despite a nonzero topk_length.
    valid = (indices[:, 0] >= 0) & (indices[:, 0] < kv.shape[0])
    valid &= torch.arange(topk, device="cuda")[None, :] < lengths[:, None]
    valid_lengths = valid.sum(dim=-1)
    sink = (
        torch.full((heads,), 3.0, device="cuda", dtype=torch.float32)
        if with_sink
        else None
    )

    def run():
        # Omit the argument for None to keep exercising the legacy default.
        kwargs = {} if return_log2_lse is None else {"return_log2_lse": return_log2_lse}
        return flash_mla_sparse_fwd(
            q, kv, indices, dim**-0.5, attn_sink=sink, topk_length=lengths, **kwargs
        )

    def check(result, q_value):
        assert isinstance(result, list)  # Preserve the native vector/list container.
        out, maximum, lse = result
        score = q_value * math.sqrt(dim) * 0.125
        expected_max = torch.full((3, heads), score, device="cuda", dtype=torch.float32)
        expected_max[valid_lengths == 0] = -torch.inf
        expected_lse = expected_max + valid_lengths.float().log()[:, None]
        expected_lse[valid_lengths == 0] = torch.inf
        if return_log2_lse is False:
            assert maximum is None
            assert lse is None
        else:
            if return_log2_lse is True:
                expected_max = expected_max / math.log(2)
                expected_lse = expected_lse / math.log(2)
            torch.testing.assert_close(maximum, expected_max, atol=2e-4, rtol=2e-4)
            torch.testing.assert_close(lse, expected_lse, atol=2e-4, rtol=2e-4)
        mass = valid_lengths.float() * math.exp(score)
        probability = torch.where(
            valid_lengths > 0,
            mass / (mass + (math.exp(3.0) if with_sink else 0.0)),
            0.0,
        )
        expected_out = (probability[:, None, None] * 0.125).expand_as(out)
        torch.testing.assert_close(out.float(), expected_out, atol=5e-4, rtol=0.01)

    check(run(), 1.0)
    if return_log2_lse is None:
        # Raw seven-argument calls must still bind and return natural-log stats.
        check(
            flash_mla_cuda.sparse_prefill_fwd(
                q, kv, indices, dim**-0.5, 512, sink, lengths
            ),
            1.0,
        )
        check(
            flash_mla_cuda.sparse_prefill_fwd(
                q, kv, indices, dim**-0.5, 512, sink, lengths, return_log2_lse=None
            ),
            1.0,
        )
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    q.fill_(2)
    graph.replay()
    torch.cuda.synchronize()
    check(result, 2.0)
