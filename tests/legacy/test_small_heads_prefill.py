"""Focused SM10x regression checks for small-head sparse MLA prefill.

Run against the newly built wheel: python tests/legacy/test_small_heads_prefill.py.
"""
import unittest

import torch

from flash_mla import flash_mla_sparse_fwd, flash_mla_sparse_fwd_small_heads


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SmallHeadsPrefillTest(unittest.TestCase):
    def test_outputs_and_graph_reuse(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("SM10x required")
        torch.manual_seed(4101)
        for heads in (8, 32):
            for dim in (512, 576):
                for topk in (128, 192, 2048):
                    with self.subTest(heads=heads, dim=dim, topk=topk):
                        q = torch.zeros(157, 64, dim, device="cuda", dtype=torch.bfloat16)
                        q[:, :heads].normal_()
                        kv = torch.randn(4096, 1, dim, device="cuda", dtype=torch.bfloat16)
                        ids = torch.randint(4096, (157, 1, topk), device="cuda", dtype=torch.int32)
                        lengths = torch.randint(topk + 1, (157,), device="cuda", dtype=torch.int32)
                        lengths[0] = 0
                        ids[1].fill_(-1)
                        sink = torch.randn(64, device="cuda")
                        sink[0], sink[1] = torch.inf, -torch.inf
                        for use_sink in (False, True):
                            sk = sink if use_sink else None
                            def run():
                                return flash_mla_sparse_fwd_small_heads(q, kv, ids, dim**-0.5, heads, sk, lengths)
                            for _ in range(2):
                                run()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                out, mx, lse = run()
                            self.assertIsNone(mx)
                            self.assertIsNone(lse)
                            self.assertEqual(out.shape, (157, heads, 512))
                            for scale in (1.0, 3.0):
                                q[:, :heads].normal_().mul_(scale)
                                graph.replay()
                                ref = flash_mla_sparse_fwd(q, kv, ids, dim**-0.5, attn_sink=sk,
                                    topk_length=lengths, use_legacy=True, return_log2_lse=False)[0]
                                torch.testing.assert_close(out, ref[:, :heads], atol=.03, rtol=.03)
                        empty = flash_mla_sparse_fwd_small_heads(q[:0], kv, ids[:0], dim**-0.5, heads)[0]
                        self.assertEqual(empty.shape, (0, heads, 512))
                        with self.assertRaises(RuntimeError):
                            flash_mla_sparse_fwd_small_heads(q, kv, ids, dim**-0.5, 16)


if __name__ == "__main__":
    unittest.main()
