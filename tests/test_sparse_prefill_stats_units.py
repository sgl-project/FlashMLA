"""The native Python API must keep natural-log statistics, unlike the SGL shim."""

import math

import pytest
import torch

from flash_mla import flash_mla_sparse_fwd

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("dim", [512, 576])
@pytest.mark.parametrize("topk", [128, 512])
@pytest.mark.parametrize("with_sink", [False, True])
def test_native_prefill_stats_units(heads, dim, topk, with_sink):
    if torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("Sparse prefill requires SM90 or SM100-family")
    q = torch.ones(3, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.full((1024, 1, dim), 0.125, device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros(3, 1, topk, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([0, 65, topk], device="cuda", dtype=torch.int32)
    sink = (
        torch.full((heads,), 3.0, device="cuda", dtype=torch.float32)
        if with_sink
        else None
    )

    def run():
        return flash_mla_sparse_fwd(
            q, kv, indices, dim**-0.5, attn_sink=sink, topk_length=lengths
        )

    def check(result, q_value):
        out, maximum, lse = result
        score = q_value * math.sqrt(dim) * 0.125
        expected_max = torch.full((3, heads), score, device="cuda", dtype=torch.float32)
        expected_max[0] = -torch.inf
        expected_lse = expected_max + lengths.float().log()[:, None]
        expected_lse[0] = torch.inf
        torch.testing.assert_close(maximum, expected_max, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(lse, expected_lse, atol=2e-4, rtol=2e-4)
        mass = lengths.float() * math.exp(score)
        probability = torch.where(
            lengths > 0,
            mass / (mass + (math.exp(3.0) if with_sink else 0.0)),
            0.0,
        )
        expected_out = (probability[:, None, None] * 0.125).expand_as(out)
        torch.testing.assert_close(out.float(), expected_out, atol=5e-4, rtol=0.01)

    check(run(), 1.0)
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
