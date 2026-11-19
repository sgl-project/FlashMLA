"""SM10x checks for sparse MLA prefill with an extra KV cache (extra_kv / extra_indices / extra_topk_length).

Run against the newly built wheel: python tests/legacy/test_sparse_prefill_extra_kv.py.
"""
import math
import unittest

import torch

from flash_mla import flash_mla_sparse_fwd, flash_mla_sparse_fwd_small_heads, has_sparse_prefill_extra_kv

SCALE = 576**-0.5


def _gather(kv, ids, length, rows):
    """Selected keys [n, topk, 576] in fp32 and their validity, honoring -1 / out-of-range slots and lengths."""
    ids = ids[rows, 0].long()
    valid = (ids >= 0) & (ids < kv.shape[0])
    if length is not None:
        valid &= torch.arange(ids.shape[1], device=ids.device)[None] < length[rows].clamp(0, ids.shape[1])[:, None]
    return kv[ids.clamp(0, kv.shape[0] - 1), 0].float(), valid


def _reference(q, kv, ids, extra_kv, extra_ids, heads, length=None, extra_length=None, sink=None, chunk=64):
    """FP32 softmax over both caches' selected keys: output, max_logits and natural-log lse per (row, head)."""
    out, max_logits, lse = [], [], []
    for start in range(0, q.shape[0], chunk):
        rows = slice(start, min(start + chunk, q.shape[0]))
        k_main, v_main = _gather(kv, ids, length, rows)
        k_extra, v_extra = _gather(extra_kv, extra_ids, extra_length, rows)
        keys, valid = torch.cat((k_main, k_extra), 1), torch.cat((v_main, v_extra), 1)
        scores = (q[rows, :heads].float() @ keys.transpose(1, 2)) * SCALE
        scores = scores.masked_fill(~valid[:, None], float("-inf"))
        row_lse = torch.logsumexp(scores, -1)
        lse_for_o = row_lse if sink is None else torch.logsumexp(torch.stack((row_lse, sink[:heads].expand_as(row_lse))), 0)
        lse_for_o = torch.where(lse_for_o == float("-inf"), torch.full_like(lse_for_o, float("inf")), lse_for_o)
        out.append(torch.exp(scores - lse_for_o[..., None]) @ keys[..., :512])
        max_logits.append(scores.max(-1).values)
        lse.append(row_lse)
    return torch.cat(out), torch.cat(max_logits), torch.cat(lse)


def _inputs(rows, topk, extra_topk, s_kv=32768, extra_s_kv=1024, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(rows, 64, 576, generator=g, device="cuda") * 0.2).to(torch.bfloat16)
    kv = (torch.randn(s_kv, 1, 576, generator=g, device="cuda") * 0.2).to(torch.bfloat16)
    extra_kv = (torch.randn(extra_s_kv, 1, 576, generator=g, device="cuda") * 0.2).to(torch.bfloat16)
    ids = torch.randint(0, s_kv, (rows, 1, topk), generator=g, device="cuda", dtype=torch.int32)
    extra_ids = torch.randint(0, extra_s_kv, (rows, 1, extra_topk), generator=g, device="cuda", dtype=torch.int32)
    # Empty caches, masked tails and an out-of-range slot
    ids[0].fill_(-1)
    if rows > 1:
        extra_ids[1].fill_(-1)
    if rows > 2:
        ids[2, 0, 31:] = -1
        extra_ids[2, 0, 17:] = -1
    if rows > 3:
        ids[3].fill_(-1)
        extra_ids[3].fill_(-1)
    if rows > 4:
        ids[4, 0, 5] = s_kv + 3
    return q, kv, ids, extra_kv, extra_ids


def _lengths(rows, topk, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    length = torch.randint(0, topk + 1, (rows,), generator=g, device="cuda", dtype=torch.int32)
    edges = torch.tensor([0, 1, 63, 64, topk, topk + 50, -5], dtype=torch.int32)[:rows]
    length[:len(edges)] = edges
    return length


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SparsePrefillExtraKvTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("SM10x required")
        self.assertTrue(has_sparse_prefill_extra_kv())
        self.many_rows = 2 * torch.cuda.get_device_properties(0).multi_processor_count + 37

    def test_small_heads(self):
        for heads in (8, 32):
            for rows in (1, 7, 157, self.many_rows):
                for topk, extra_topk in ((2048, 128), (128, 192)):
                    for with_lengths in (False, True):
                        with self.subTest(heads=heads, rows=rows, topk=topk, extra_topk=extra_topk,
                                          with_lengths=with_lengths):
                            q, kv, ids, extra_kv, extra_ids = _inputs(rows, topk, extra_topk, seed=rows + heads)
                            length = _lengths(rows, topk, 1) if with_lengths else None
                            extra_length = _lengths(rows, extra_topk, 2) if with_lengths else None
                            sink = torch.randn(64, device="cuda") if with_lengths else None
                            out = flash_mla_sparse_fwd_small_heads(
                                q, kv, ids, SCALE, heads, sink, length,
                                extra_kv=extra_kv, extra_indices=extra_ids, extra_topk_length=extra_length)[0]
                            self.assertEqual(out.shape, (rows, heads, 512))
                            ref = _reference(q, kv, ids, extra_kv, extra_ids, heads, length, extra_length, sink)[0]
                            torch.testing.assert_close(out.float(), ref, atol=0.002, rtol=0.03)

    def test_64_heads_with_statistics(self):
        for rows in (1, 157, self.many_rows):
            for with_lengths in (False, True):
                with self.subTest(rows=rows, with_lengths=with_lengths):
                    q, kv, ids, extra_kv, extra_ids = _inputs(rows, 2048, 128, seed=rows)
                    length = _lengths(rows, 2048, 3) if with_lengths else None
                    extra_length = _lengths(rows, 128, 4) if with_lengths else None
                    sink = torch.randn(64, device="cuda")
                    sink[0], sink[1] = float("inf"), float("-inf")
                    out, max_logits, lse = flash_mla_sparse_fwd(
                        q, kv, ids, SCALE, attn_sink=sink, topk_length=length, use_legacy=True,
                        extra_kv=extra_kv, extra_indices=extra_ids, extra_topk_length=extra_length)
                    ref_out, ref_max, ref_lse = _reference(q, kv, ids, extra_kv, extra_ids, 64, length, extra_length, sink)
                    torch.testing.assert_close(out.float(), ref_out, atol=0.002, rtol=0.03)
                    has_keys = torch.isfinite(ref_lse)
                    torch.testing.assert_close(lse[has_keys], ref_lse[has_keys], atol=1e-3, rtol=1e-3)
                    torch.testing.assert_close(max_logits[has_keys], ref_max[has_keys], atol=1e-3, rtol=1e-3)
                    _, max_logits2, lse2 = flash_mla_sparse_fwd(
                        q, kv, ids, SCALE, attn_sink=sink, topk_length=length, use_legacy=True, return_log2_lse=True,
                        extra_kv=extra_kv, extra_indices=extra_ids, extra_topk_length=extra_length)
                    torch.testing.assert_close(lse2[has_keys], lse[has_keys] * math.log2(math.e), atol=1e-3, rtol=1e-4)
                    torch.testing.assert_close(max_logits2[has_keys], max_logits[has_keys] * math.log2(math.e),
                                               atol=1e-3, rtol=1e-4)

    def test_graph_replay(self):
        q, kv, ids, extra_kv, extra_ids = _inputs(157, 2048, 128, seed=9)

        def run():
            return flash_mla_sparse_fwd_small_heads(q, kv, ids, SCALE, 8, extra_kv=extra_kv, extra_indices=extra_ids)[0]

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = run()
        q.normal_(std=0.2)
        graph.replay()
        torch.testing.assert_close(replayed, run(), atol=0, rtol=0)

    def test_empty_query_and_invalid_arguments(self):
        q, kv, ids, extra_kv, extra_ids = _inputs(4, 2048, 128)
        empty = flash_mla_sparse_fwd_small_heads(q[:0], kv, ids[:0], SCALE, 8, extra_kv=extra_kv,
                                                 extra_indices=extra_ids[:0])[0]
        self.assertEqual(empty.shape, (0, 8, 512))
        lengths = torch.zeros(4, device="cuda", dtype=torch.int32)
        for bad in (
            dict(extra_indices=None),
            dict(extra_kv=None, extra_topk_length=lengths),
            dict(extra_indices=extra_ids[:, :, :64]),
            dict(extra_indices=extra_ids.long()),
            dict(extra_kv=extra_kv.float()),
            dict(extra_topk_length=lengths[:3]),
            dict(extra_topk_length=lengths.long()),
            dict(active_heads=16),
        ):
            args = dict(extra_kv=extra_kv, extra_indices=extra_ids, extra_topk_length=None, active_heads=8) | bad
            with self.subTest(bad=list(bad)), self.assertRaises(RuntimeError):
                flash_mla_sparse_fwd_small_heads(q, kv, ids, SCALE, args["active_heads"], extra_kv=args["extra_kv"],
                                                 extra_indices=args["extra_indices"],
                                                 extra_topk_length=args["extra_topk_length"])
        with self.assertRaises(ValueError):
            flash_mla_sparse_fwd(q, kv, ids, SCALE, extra_kv=extra_kv, extra_indices=extra_ids)


if __name__ == "__main__":
    unittest.main()
