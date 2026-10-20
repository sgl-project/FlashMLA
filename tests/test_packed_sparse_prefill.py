"""Correctness only: no timing loops or performance assertions."""

import pytest
import torch

from flash_mla.packed_sparse_prefill import flash_mla_packed_sparse_fwd


@pytest.fixture(autouse=True)
def require_supported_blackwell():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    prop = torch.cuda.get_device_properties(0)
    if (prop.major, prop.minor, prop.multi_processor_count) not in (
        (10, 0, 148),
        (10, 3, 148),
    ):
        pytest.skip("Packing is qualified on 148-SM B200 and B300 GPUs")


def inputs(rows, width, padded=True):
    torch.manual_seed(419)
    q = torch.randn(
        rows, 64 if padded else 16, 512, device="cuda", dtype=torch.bfloat16
    )
    kv = torch.randn(2048, 1, 512, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, 2048, (rows, width), device="cuda", dtype=torch.int32)
    lengths = torch.full((rows,), width, device="cuda", dtype=torch.int32)
    # Empty rows, repeated keys, overlapping queries, and invalid key positions.
    indices[0] = -1
    if rows > 1:
        indices[1, :8] = 0
        indices[1, 8:12] = 2048
    if rows > 2:
        indices[2] = indices[1]
        lengths[2] = width // 2
    if rows > 3:
        lengths[3] = 0
    sink = torch.randn(16, device="cuda", dtype=torch.float32)
    return q[:, :16], kv, indices, lengths, sink


def assert_reference(result, q, kv, indices, lengths, scale, sink):
    out, max_logits, lse = result
    rows = q.shape[0]
    assert out.shape == (rows, 16, 512)
    assert max_logits.shape == lse.shape == (rows, 16)
    assert out.dtype == torch.bfloat16
    assert max_logits.dtype == lse.dtype == torch.float32
    assert torch.isfinite(out).all()
    # Includes both halves of a packed tile and the final partial tile.
    samples = sorted(
        {
            0,
            min(1, rows - 1),
            min(2, rows - 1),
            min(3, rows - 1),
            rows // 2,
            rows - 1,
        }
    )
    for row in samples:
        idx = indices[row, : int(lengths[row])].long()
        idx = idx[(idx >= 0) & (idx < kv.shape[0])]
        if idx.numel() == 0:
            assert torch.count_nonzero(out[row]) == 0
            assert torch.isneginf(max_logits[row]).all()
            assert torch.isposinf(lse[row]).all()
            continue
        values = kv[idx, 0].float()
        logits = (q[row].float() @ values.T) * scale
        reference_max = logits.max(-1).values
        reference_lse = logits.logsumexp(-1)
        denominator = (
            reference_lse if sink is None else torch.logaddexp(reference_lse, sink)
        )
        expected = (logits - denominator[:, None]).exp() @ values
        torch.testing.assert_close(out[row].float(), expected, atol=0.025, rtol=0.025)
        torch.testing.assert_close(
            max_logits[row], reference_max, atol=0.01, rtol=0.005
        )
        torch.testing.assert_close(lse[row], reference_lse, atol=0.01, rtol=0.005)


@pytest.mark.parametrize(
    "rows,width",
    [
        (1, 128),
        (3, 640),
        (4, 128),
        (5, 640),
        (511, 128),
        (512, 128),
        (513, 128),
        (4095, 640),
        (4096, 640),
        (4097, 640),
        (4160, 640),
    ],
)
@pytest.mark.parametrize("padded", [False, True])
def test_packed_reference(rows, width, padded):
    q, kv, indices, lengths, sink = inputs(rows, width, padded)
    scale = 512**-0.5
    result = flash_mla_packed_sparse_fwd(q, kv, indices, lengths, scale, sink)
    assert_reference(result, q, kv, indices, lengths, scale, sink)


@pytest.mark.parametrize("sink_enabled", [False, True])
@pytest.mark.parametrize("scale", [0.0, 512**-0.5])
def test_small_edge_cases(sink_enabled, scale):
    q, kv, indices, lengths, sink = inputs(5, 640)
    sink = sink if sink_enabled else None
    result = flash_mla_packed_sparse_fwd(q, kv, indices, lengths, scale, sink)
    assert_reference(result, q, kv, indices, lengths, scale, sink)


def test_graph_replay():
    q, kv, indices, lengths, sink = inputs(513, 640)
    scale = 512**-0.5
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        flash_mla_packed_sparse_fwd(q, kv, indices, lengths, scale, sink)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = flash_mla_packed_sparse_fwd(q, kv, indices, lengths, scale, sink)
    for key in (17, 31):
        indices.fill_(key)
        lengths.fill_(64)
        graph.replay()
        assert_reference(result, q, kv, indices, lengths, scale, sink)


def test_reject_padded_head_count():
    q, kv, indices, lengths, sink = inputs(5, 128)
    q64 = torch.empty(5, 64, 512, device="cuda", dtype=q.dtype)
    with pytest.raises(RuntimeError, match="16, 512"):
        flash_mla_packed_sparse_fwd(q64, kv, indices, lengths, 512**-0.5, sink)
