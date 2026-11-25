"""SM10x checks for sparse MLA prefill with token packing."""
import unittest

import torch

from flash_mla import (
    flash_mla_dequant_v32_kv,
    flash_mla_sparse_fwd,
    flash_mla_sparse_fwd_packed,
    flash_mla_sparse_pack_indices,
    has_sparse_prefill_token_packing,
)

SCALE = 576**-0.5
GROUP_SHIFT = 26
SLOT_MASK = (1 << GROUP_SHIFT) - 1
MAX_GROUPS = 5


def _group_size(heads):
    return min(64 // heads, MAX_GROUPS)


def _reference(q, kv, ids, length=None, chunk=64):
    """FP32 softmax over each token's own selected keys."""
    out = []
    for start in range(0, q.shape[0], chunk):
        rows = slice(start, min(start + chunk, q.shape[0]))
        cur = ids[rows].long()
        valid = (cur >= 0) & (cur < kv.shape[0])
        if length is not None:
            valid &= torch.arange(cur.shape[1], device=cur.device)[None] < length[rows].clamp(0, cur.shape[1])[:, None]
        keys = kv[cur.clamp(0, kv.shape[0] - 1), 0].float()
        scores = (q[rows].float() @ keys.transpose(1, 2)) * SCALE
        scores = scores.masked_fill(~valid[:, None], float("-inf"))
        lse = torch.logsumexp(scores, -1)
        lse = torch.where(lse == float("-inf"), torch.full_like(lse, float("inf")), lse)
        out.append(torch.exp(scores - lse[..., None]) @ keys[..., :512])
    return torch.cat(out)


def _inputs(tokens, heads, topk, s_kv, seed, shared=0.6):
    """Index lists that are distinct within a row and overlap between neighbouring tokens."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(tokens, heads, 576, generator=g, device="cuda").to(torch.bfloat16)
    kv = torch.randn(s_kv, 1, 576, generator=g, device="cuda").to(torch.bfloat16)
    ids = torch.empty(tokens, topk, device="cuda", dtype=torch.int32)
    window = min(s_kv, 2 * topk)
    for t in range(tokens):
        order = torch.randperm(s_kv, generator=g, device="cuda")
        start = (t * 7) % (s_kv - window + 1)
        in_window = (order >= start) & (order < start + window)
        near, far = order[in_window], order[~in_window]
        n_near = min(int(topk * shared), near.numel())
        ids[t] = torch.cat((near[:n_near], far[: topk - n_near]))[torch.randperm(topk, generator=g, device="cuda")]
    return q, kv, ids


def _add_edge_rows(ids, s_kv):
    """Empty rows, masked tails and an out-of-range slot. Returns per-token lengths."""
    tokens, topk = ids.shape
    if tokens > 0:
        ids[0].fill_(-1)
    if tokens > 2:
        ids[2, 31:] = -1
    if tokens > 3:
        ids[3, 1:] = -1
    if tokens > 4:
        ids[4, 5] = s_kv + 3
    if tokens > 7:
        ids[7, topk - 64:] = -1
    return (ids >= 0).sum(1, dtype=torch.int32)


def _run_packed(q, kv, ids, lengths, heads, packed=None, pack_lengths=None):
    """T // G groups through the packed kernel, the T % G rows left over through the unpacked one."""
    tokens = q.shape[0]
    group = _group_size(heads)
    n = tokens // group
    out = torch.empty(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    if n:
        packed, pack_lengths = flash_mla_sparse_pack_indices(ids[: n * group], group, packed, pack_lengths)
        width = max(128, (group * ids.shape[1] + 63) // 64 * 64)
        flash_mla_sparse_fwd_packed(
            q[: n * group].view(n, group * heads, 576), kv, packed[:n, :width].unsqueeze(1), SCALE, heads,
            pack_lengths[:n], out[: n * group].view(n, group * heads, 512),
        )
    if n * group < tokens:
        flash_mla_sparse_fwd_packed(
            q[n * group:], kv, ids[n * group:].unsqueeze(1), SCALE, 0, lengths[n * group:], out[n * group:]
        )
    return out


def _rel_l2(out, ref):
    return float((out.float() - ref).norm() / ref.norm().clamp_min(1e-12))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SparsePrefillTokenPackingTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("SM10x required")
        self.assertTrue(has_sparse_prefill_token_packing())
        self.many_rows = 2 * torch.cuda.get_device_properties(0).multi_processor_count + 37

    def _check_pack(self, ids, group, packed, lengths):
        tokens, topk = ids.shape
        n = tokens // group
        self.assertEqual(lengths.shape[0], n)
        cpu_ids, cpu_packed, cpu_lengths = ids.cpu(), packed.cpu(), lengths.cpu()
        for g in range(n):
            want = {}
            for j in range(group):
                for slot in cpu_ids[g * group + j].tolist():
                    if slot >= 0:
                        want[slot] = want.get(slot, 0) | (1 << j)
            length = int(cpu_lengths[g])
            self.assertEqual(length, len(want))
            row = cpu_packed[g].tolist()
            got = {entry & SLOT_MASK: entry >> GROUP_SHIFT for entry in row[:length]}
            self.assertEqual(len(got), length, "a slot appears twice")
            self.assertEqual(got, want)
            pad_end = max(128, (length + 63) // 64 * 64)
            self.assertTrue(all(entry == -1 for entry in row[length:pad_end]))

    def test_pack_indices(self):
        for group in (1, 2, 4, 5):
            for topk in (64, 128, 2048):
                with self.subTest(group=group, topk=topk):
                    _, _, ids = _inputs(4 * group, 12, topk, 8192, seed=100 * group + topk)
                    # Duplicates within a row, an empty row, a masked tail, slots that every row of a group selects
                    ids[1, 3] = ids[1, 2]
                    ids[1, topk - 1] = ids[1, 2]
                    ids[group].fill_(-1)
                    ids[2 * group, topk // 2:] = -1
                    ids[3 * group: 4 * group, :8] = torch.arange(8, device="cuda", dtype=torch.int32) + 4000
                    packed, lengths = flash_mla_sparse_pack_indices(ids, group)
                    self.assertEqual(packed.shape, (4, max(128, (group * topk + 63) // 64 * 64)))
                    self._check_pack(ids, group, packed, lengths)

                    # Same lists for any entry order and any workspace contents
                    shuffled = ids[:, torch.randperm(topk, device="cuda")].contiguous()
                    again, again_lengths = flash_mla_sparse_pack_indices(shuffled, group)
                    workspace = torch.full((6, packed.shape[1] + 64), 12345, device="cuda", dtype=torch.int32)
                    workspace_lengths = torch.full((9,), -7, device="cuda", dtype=torch.int32)
                    flash_mla_sparse_pack_indices(ids, group, workspace, workspace_lengths)
                    self.assertTrue(torch.equal(lengths, again_lengths))
                    self.assertTrue(torch.equal(lengths, workspace_lengths[:4]))
                    for g in range(4):
                        pad_end = max(128, (int(lengths[g]) + 63) // 64 * 64)
                        self.assertTrue(torch.equal(packed[g, :pad_end], again[g, :pad_end]))
                        self.assertTrue(torch.equal(packed[g, :pad_end], workspace[g, :pad_end]))

    def test_pack_indices_all_empty_and_no_rows(self):
        ids = torch.full((10, 128), -1, device="cuda", dtype=torch.int32)
        packed, lengths = flash_mla_sparse_pack_indices(ids, 5)
        self.assertTrue(bool((lengths == 0).all()))
        self.assertTrue(bool((packed[:, :128] == -1).all()))
        packed, lengths = flash_mla_sparse_pack_indices(ids[:0], 5)
        self.assertEqual(packed.shape[0], 0)
        self.assertEqual(lengths.shape[0], 0)

    def test_pack_indices_invalid_arguments(self):
        ids = torch.zeros((10, 128), device="cuda", dtype=torch.int32)
        with self.assertRaises(RuntimeError):
            flash_mla_sparse_pack_indices(ids, 6)
        with self.assertRaises(RuntimeError):
            flash_mla_sparse_pack_indices(ids, 4)   # 10 rows are not a multiple of 4
        with self.assertRaises(RuntimeError):
            flash_mla_sparse_pack_indices(ids, 5, torch.empty((2, 512), device="cuda", dtype=torch.int32))
        with self.assertRaises(RuntimeError):
            flash_mla_sparse_pack_indices(torch.zeros((5, 4096), device="cuda", dtype=torch.int32), 5)

    def test_dequant_v32(self):
        g = torch.Generator(device="cuda").manual_seed(7)
        pages = 9
        kv = torch.empty(pages, 64, 1, 656, device="cuda", dtype=torch.uint8)
        nope = (torch.randn(pages, 64, 512, generator=g, device="cuda") * 2).to(torch.float8_e4m3fn)
        scale = torch.rand(pages, 64, 4, generator=g, device="cuda") * 0.1 + 0.03
        rope = torch.randn(pages, 64, 64, generator=g, device="cuda").to(torch.bfloat16)
        kv[:, :, 0, :512] = nope.view(torch.uint8)
        kv[:, :, 0, 512:528] = scale.contiguous().view(torch.uint8).view(pages, 64, 16)
        kv[:, :, 0, 528:] = rope.contiguous().view(torch.uint8).view(pages, 64, 128)
        want = torch.cat(
            ((nope.float() * scale.repeat_interleave(128, dim=-1)).to(torch.bfloat16), rope), -1
        ).view(pages * 64, 1, 576)

        out = flash_mla_dequant_v32_kv(kv)
        self.assertEqual(out.shape, (pages * 64, 1, 576))
        self.assertTrue(torch.equal(out.view(torch.int16), want.view(torch.int16)))

        rows = torch.randperm(pages * 64, generator=g, device="cuda")[:333].int()
        out = flash_mla_dequant_v32_kv(kv, rows=rows)
        self.assertTrue(torch.equal(out.view(torch.int16), want[rows.long()].view(torch.int16)))

        # A caller's buffer shorter than the cache, FP8-typed pages, and no rows at all
        part = torch.empty(100, 1, 576, device="cuda", dtype=torch.bfloat16)
        flash_mla_dequant_v32_kv(kv.view(torch.float8_e4m3fn), out=part)
        self.assertTrue(torch.equal(part.view(torch.int16), want[:100].view(torch.int16)))
        self.assertEqual(flash_mla_dequant_v32_kv(kv, out=part[:0]).shape, (0, 1, 576))

        with self.assertRaises(RuntimeError):
            flash_mla_dequant_v32_kv(kv.view(pages * 64, 1, 656))
        with self.assertRaises(RuntimeError):
            flash_mla_dequant_v32_kv(kv, out=torch.empty(pages * 64 + 1, 1, 576, device="cuda", dtype=torch.bfloat16))

    def test_packed_matches_reference_and_unpacked(self):
        for heads in (8, 12, 16, 32):
            group = _group_size(heads)
            for topk in (128, 192, 2048):
                for tokens in (group * 31 + (group - 1), group * 8):   # with and without rows left over
                    with self.subTest(heads=heads, topk=topk, tokens=tokens):
                        s_kv = 32768
                        q, kv, ids = _inputs(tokens, heads, topk, s_kv, seed=heads * 10000 + topk + tokens)
                        lengths = _add_edge_rows(ids, s_kv)
                        ref = _reference(q, kv, ids, lengths)

                        out = _run_packed(q, kv, ids, lengths, heads)
                        self.assertTrue(bool(torch.isfinite(out).all()))
                        torch.testing.assert_close(out.float(), ref, atol=.03, rtol=.03)
                        self.assertLess(_rel_l2(out, ref), 1e-2)
                        self.assertEqual(int(out[0].abs().max()), 0)   # a token with no key at all

                        # Unpacked call, and the existing kernel on Q padded to 64 heads
                        plain = flash_mla_sparse_fwd_packed(q, kv, ids.unsqueeze(1), SCALE, 0, lengths)
                        self.assertEqual(plain.shape, (tokens, heads, 512))
                        torch.testing.assert_close(plain.float(), ref, atol=.03, rtol=.03)
                        q64 = q.new_zeros((tokens, 64, 576))
                        q64[:, :heads] = q
                        padded = flash_mla_sparse_fwd(q64, kv, ids.unsqueeze(1), SCALE, topk_length=lengths,
                                                      use_legacy=True, return_log2_lse=False)[0]
                        torch.testing.assert_close(plain, padded[:, :heads], atol=.03, rtol=.03)
                        torch.testing.assert_close(out, padded[:, :heads], atol=.03, rtol=.03)

                        # Deterministic through a reused workspace
                        width = max(128, (group * topk + 63) // 64 * 64)
                        workspace = torch.full((tokens // group + 3, width + 128), 777, device="cuda", dtype=torch.int32)
                        workspace_lengths = torch.zeros(tokens // group + 3, device="cuda", dtype=torch.int32)
                        again = _run_packed(q, kv, ids, lengths, heads, workspace, workspace_lengths)
                        self.assertTrue(torch.equal(out, again))

    def test_partial_group(self):
        # Missing tokens of the last group are zero rows of Q with empty index lists
        for heads in (12, 32):
            group = _group_size(heads)
            for real in range(1, group):
                with self.subTest(heads=heads, real_tokens_in_last_group=real):
                    tokens = group * 3 + real
                    q, kv, ids = _inputs(tokens, heads, 192, 8192, seed=heads * 100 + real)
                    lengths = (ids >= 0).sum(1, dtype=torch.int32)
                    ref = _reference(q, kv, ids, lengths)
                    n = 4
                    q_full = q.new_zeros((n * group, heads, 576))
                    q_full[:tokens] = q
                    ids_full = ids.new_full((n * group, 192), -1)
                    ids_full[:tokens] = ids
                    packed, pack_lengths = flash_mla_sparse_pack_indices(ids_full, group)
                    self.assertTrue(bool((packed[n - 1, :int(pack_lengths[n - 1])] >> (GROUP_SHIFT + real) == 0).all()))
                    out = flash_mla_sparse_fwd_packed(
                        q_full.view(n, group * heads, 576), kv, packed.unsqueeze(1), SCALE, heads, pack_lengths
                    ).view(n * group, heads, 512)
                    torch.testing.assert_close(out[:tokens].float(), ref, atol=.03, rtol=.03)
                    self.assertEqual(int(out[tokens:].abs().max()), 0)

    def test_head_counts_that_do_not_fill_the_tile(self):
        for heads, group in ((12, 5), (24, 2), (12, 3), (20, 3), (64, 1)):
            with self.subTest(heads=heads, group=group):
                tokens = group * 9
                q, kv, ids = _inputs(tokens, heads, 256, 8192, seed=heads * 7 + group)
                lengths = _add_edge_rows(ids, 8192)
                ref = _reference(q, kv, ids, lengths)
                packed, pack_lengths = flash_mla_sparse_pack_indices(ids, group)
                out = flash_mla_sparse_fwd_packed(
                    q.view(9, group * heads, 576), kv, packed.unsqueeze(1), SCALE, heads, pack_lengths
                ).view(tokens, heads, 512)
                torch.testing.assert_close(out.float(), ref, atol=.03, rtol=.03)

    def test_disjoint_and_identical_lists(self):
        heads, group, topk = 12, 5, 128
        tokens = group * 6
        q, kv, _ = _inputs(tokens, heads, topk, 8192, seed=11)
        disjoint = torch.arange(tokens * topk, device="cuda", dtype=torch.int32).view(tokens, topk) % 8192
        same = torch.arange(topk, device="cuda", dtype=torch.int32).repeat(tokens, 1) + 17
        for name, ids, union in (("disjoint", disjoint, group * topk), ("identical", same, topk)):
            with self.subTest(lists=name):
                packed, pack_lengths = flash_mla_sparse_pack_indices(ids, group)
                self.assertTrue(bool((pack_lengths == union).all()))
                out = flash_mla_sparse_fwd_packed(
                    q.view(6, group * heads, 576), kv, packed.unsqueeze(1), SCALE, heads, pack_lengths
                ).view(tokens, heads, 512)
                torch.testing.assert_close(out.float(), _reference(q, kv, ids), atol=.03, rtol=.03)

    def test_several_jobs_per_cta_and_graph_replay(self):
        # More groups than SMs, so CTAs fetch further rows by cluster launch control
        heads, group, topk = 12, 5, 2048
        tokens = self.many_rows * group + 3
        q, kv, ids = _inputs(tokens, heads, topk, 32768, seed=5)
        lengths = _add_edge_rows(ids, 32768)
        n = tokens // group
        width = (group * topk + 63) // 64 * 64
        packed = torch.empty(n, width, device="cuda", dtype=torch.int32)
        pack_lengths = torch.empty(n, device="cuda", dtype=torch.int32)

        def run():
            return _run_packed(q, kv, ids, lengths, heads, packed, pack_lengths)

        eager = run()
        torch.testing.assert_close(eager.float(), _reference(q, kv, ids, lengths), atol=.03, rtol=.03)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
        for scale in (1.0, 3.0):
            q.normal_().mul_(scale)
            graph.replay()
            torch.testing.assert_close(out.float(), _reference(q, kv, ids, lengths), atol=.03, rtol=.03)

    def test_empty_input_and_invalid_arguments(self):
        q, kv, ids = _inputs(10, 12, 128, 4096, seed=3)
        packed, pack_lengths = flash_mla_sparse_pack_indices(ids, 5)
        self.assertEqual(flash_mla_sparse_fwd_packed(q[:0], kv, ids[:0].unsqueeze(1), SCALE, 0).shape, (0, 12, 512))
        empty = flash_mla_sparse_fwd_packed(q[:0].view(0, 60, 576), kv, packed[:0].unsqueeze(1), SCALE, 12, pack_lengths[:0])
        self.assertEqual(empty.shape, (0, 60, 512))
        q60 = q.view(2, 60, 576)
        with self.assertRaises(RuntimeError):   # 60 rows are more than 5 groups of 8 heads
            flash_mla_sparse_fwd_packed(q60, kv, packed.unsqueeze(1), SCALE, 8, pack_lengths)
        with self.assertRaises(RuntimeError):   # top-k width not divisible by 64
            flash_mla_sparse_fwd_packed(q60, kv, packed[:, :600].unsqueeze(1), SCALE, 12, pack_lengths)
        with self.assertRaises(RuntimeError):   # output of the wrong shape
            flash_mla_sparse_fwd_packed(q60, kv, packed.unsqueeze(1), SCALE, 12, pack_lengths,
                                        torch.empty(2, 64, 512, device="cuda", dtype=torch.bfloat16))
        with self.assertRaises(RuntimeError):   # d_qk 512 is not instantiated
            flash_mla_sparse_fwd_packed(q60[..., :512], kv[..., :512], packed.unsqueeze(1), SCALE, 12, pack_lengths)
        with self.assertRaises(RuntimeError):   # more than 64 heads
            flash_mla_sparse_fwd_packed(q.new_zeros((2, 65, 576)), kv, packed.unsqueeze(1), SCALE, 13, pack_lengths)


if __name__ == "__main__":
    unittest.main()
