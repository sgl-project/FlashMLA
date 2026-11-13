"""Tests of the fp8 sparse prefill of fused_norm_rope_attn_rope_cast.prefill (a float8_e4m3fn `kv`, h_q = 64,
enable_q_norm False: the fp8 kv with enable_q_norm True is rejected, `test_fp8_prefill_rejects_q_norm`).

Two references:
- fp64 attention on the TRUE kv values (the bf16 workspace a caller quantizes from), not on the e4m3 tensor handed to
  the kernel, so range losses of the fp8 kv (saturation above 448, subnormals) show up in the error. It does not
  implement RoPE, so these cases run with the identity RoPE (cos = 1, sin = 0); the bf16 op of the same tree (`kv` in
  bfloat16) runs on the same inputs as a second reference.
- the bf16 op of the same tree, for a random cos_sin_cache with random token_positions (`test_fp8_prefill_rope`). That
  covers the fp8 kernel's own mapping of Q tiles to rails (RoPE is applied by tile index) and the inverse RoPE of O's
  last 64 dims.

The fp8 kv is the bf16 workspace cast to e4m3 directly (unit scale; the kernel also casts Q with unit scale, after
RoPE, and scales P by 4 before its e4m3 cast). Contract: |q| and |kv| within e4m3's normal range; inputs outside it
are not tested. Distributions: `bench` (randn * 0.1), `small` (randn * 0.01, mostly e4m3 subnormals), `unselected`
(a few outlier tokens in the kv that no query selects).

Needs an SM100 GPU and the compiled flash_mla.  python -m pytest tests/test_fp8_prefill.py
"""
import math

import pytest
import torch

import flash_mla

H_Q, D, D_ROPE, N_WV = 64, 512, 64, 8
RMS_NORM_EPS = 1e-6

# Limits, set at ~1.3-1.5x the values measured on B200 (the kernels are deterministic, so the measured values repeat
# bit for bit): rel_l2 of O at ~1.33x, the min per-token cosine at ~1.45x its distance from 1, the mean |lse| error at
# ~1.45x. The comments give the measured worst case over topk 128 / 640.
# vs the fp64 reference on the true values (identity RoPE, no Q norm). The bf16 op is at its own per-32 e4m3 output cast
# (the fp8 op has it too); e4m3 Q/K/V/P add ~3.5% on `bench` / `unselected` and ~5.7% on the mostly subnormal `small`.
# `bench` attends almost uniformly, so a worse cast moves it little (round-toward-zero instead of RN for kv: rel_l2
# 0.053); `small` and the sharp RoPE case below are the cases that catch such cast regressions
FP64_LIMITS = {
    "bf16": {"rel_l2": 0.035, "tok_cos": 0.9994, "lse_mean": 5e-7},          # 0.0266, 0.99955, 3.3e-7
    "fp8 bench": {"rel_l2": 0.06, "tok_cos": 0.9984, "lse_mean": 4e-5},     # 0.0448, 0.99885, 2.7e-5
    "fp8 unselected": {"rel_l2": 0.06, "tok_cos": 0.9984, "lse_mean": 4e-5},  # 0.0448, 0.99887, 2.7e-5
    "fp8 small": {"rel_l2": 0.085, "tok_cos": 0.9963, "lse_mean": 6.5e-6},  # 0.0626, 0.99748, 4.4e-6; kv rtz: 0.111
}
MAX_LOGITS_MEAN_LIMIT = 4.5e-4    # fp8 `bench` with a sink, mean |max_logits error|: 3.0e-4

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the fused prefill needs an SM100f GPU",
)


def make_case(dist, s_q, s_kv, topk, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn(s_q, H_Q, D, generator=g) * 0.1).to(torch.bfloat16)
    std = {"bench": 0.1, "small": 0.01, "unselected": 0.1}[dist]
    kv = torch.randn(s_kv, D, generator=g) * std
    n_sel = s_kv
    if dist == "unselected":
        n_sel = s_kv - 16
        kv[n_sel:] *= 1e4     # amax ~ 4e3: outside every query's selection
    ws = kv.to(torch.bfloat16)
    idx = torch.stack([torch.randperm(n_sel, generator=g)[:topk] for _ in range(s_q)]).to(torch.int32)
    sink = torch.randn(H_Q, generator=g)
    return q, ws, idx, sink


def make_rope_case(s_q, s_kv, topk, seed):
    """Q with a per-(token, head) magnitude between 0.5 and 2 (|q| stays below ~12, inside e4m3's normal range), a kv
    like `bench`, a random RoPE, and an sm_scale that gives the scores a std of ~2.5 at unit magnitude: sharp enough
    that O depends on every 64-dim tile of Q"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    mag = torch.exp2(torch.rand(s_q, H_Q, 1, generator=g) * 2 - 1)
    q = (torch.randn(s_q, H_Q, D, generator=g) * mag).to(torch.bfloat16)
    ws = (torch.randn(s_kv, D, generator=g) * 0.1).to(torch.bfloat16)
    idx = torch.stack([torch.randperm(s_kv, generator=g)[:topk] for _ in range(s_q)]).to(torch.int32)
    sink = torch.randn(H_Q, generator=g)
    return q, ws, idx, sink, random_rope(s_q, 4096, g), D ** -0.5 * 25


def reference(q, kv_true, idx, sink, sm_scale):
    """fp64 sparse attention with sink: O [s_q, H, D], lse [s_q, H] (natural log, without the sink)"""
    q = q.double().cuda(); kv = kv_true.double().cuda(); idx = idx.long().cuda(); sink = sink.double().cuda()
    K = kv[idx]
    lg = torch.einsum("shd,std->sht", q, K) * sm_scale
    lse = torch.logsumexp(lg, -1)
    den = torch.logaddexp(lse, sink[None, :])
    return torch.einsum("sht,std->shd", torch.exp(lg - den[..., None]), K), lse


def make_partial_case(s_q, s_kv, topk, seed):
    """`bench` Q / kv with blocks of B_TOPK = 64 indices that are only partly valid. Per row: a topk_length that is not
    a multiple of 64 (1/8 of the rows below 64), so the last block ends mid-block; on the odd rows also ~30% of the
    indices before it invalid (-1, s_kv, or larger, up to 2^31 - 1), on the even rows none, so jobs mix all-valid and
    partly valid blocks; valid indices of unrelated tokens after topk_length (they must be ignored); on every 8th row
    (odd) with topk_length > 64 the whole first block invalid (the row max starts at -inf). Rows 3, 3 + 13, ... have
    every index invalid and rows 5, 5 + 29, ... topk_length 0 (lse +inf, max_logits -inf, O zero)"""
    q, ws, idx, sink = make_case("bench", s_q, s_kv, topk, seed)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    tl = torch.randint(1, topk + 1, (s_q,), generator=g)
    tl[::8] = torch.randint(1, 64, (len(tl[::8]),), generator=g)
    tl = torch.where(tl % 64 == 0, tl - 1, tl)
    pos = torch.arange(topk)[None]
    rate = 0.3 * (torch.arange(s_q) % 2)[:, None]
    bad = (torch.rand(s_q, topk, generator=g) < rate) & (pos < tl[:, None])
    bad[1::8, :64] |= tl[1::8, None] > 64
    bad[3::13] = True
    junk = torch.tensor([-1, s_kv, s_kv + 1, 2 ** 31 - 1], dtype=torch.int32)
    idx = torch.where(bad, junk[torch.randint(0, len(junk), (s_q, topk), generator=g)], idx)
    tl[5::29] = 0
    return q, ws, idx, sink, tl.int()


def reference_masked(q, kv_true, idx, sink, sm_scale, topk_length):
    """reference() with the op's validity rule: entry j of row i counts iff j < topk_length[i] and 0 <= idx < s_kv.
    Rows without a valid entry: O 0, lse and max_logits -inf (the op returns lse +inf). Returns O, lse, max_logits"""
    q = q.double().cuda(); kv = kv_true.double().cuda(); idx = idx.long().cuda(); sink = sink.double().cuda()
    pos = torch.arange(idx.shape[1], device="cuda")[None]
    valid = (pos < topk_length.cuda()[:, None]) & (idx >= 0) & (idx < kv.shape[0])
    K = kv[idx.clamp(0, kv.shape[0] - 1)]
    lg = (torch.einsum("shd,std->sht", q, K) * sm_scale).masked_fill(~valid[:, None, :], float("-inf"))
    lse = torch.logsumexp(lg, -1)
    den = torch.logaddexp(lse, sink[None, :])
    return torch.einsum("sht,std->shd", torch.exp(lg - den[..., None]).nan_to_num(0.0), K), lse, lg.amax(-1)


def identity_rope(s_q):
    """(token_positions, cos_sin_cache) of the identity RoPE"""
    return torch.zeros(s_q, dtype=torch.int32), torch.cat([torch.ones(1, D_ROPE // 2), torch.zeros(1, D_ROPE // 2)], 1)


def random_rope(s_q, n_pos, g):
    """random token_positions into a cos_sin_cache of n_pos rows of rotations by random angles"""
    theta = torch.rand(n_pos, D_ROPE // 2, generator=g, dtype=torch.float64) * (2 * math.pi)
    pos = torch.randint(0, n_pos, (s_q,), generator=g, dtype=torch.int32)
    return pos, torch.cat([theta.cos(), theta.sin()], 1).float()


def rope_tile(x, rope, tile, inverse=False):
    """x [s_q, H, D] float with the 64 dims of `tile` rotated like the kernel's RoPE (dims 2i, 2i + 1 by angle i of the
    token's cos_sin_cache row), or back"""
    pos, cs = rope
    cs = cs.to(x.device, x.dtype)[pos.to(x.device).long()]
    c = cs[:, None, :D_ROPE // 2]
    s = cs[:, None, D_ROPE // 2:] * (-1 if inverse else 1)
    lo = tile * D_ROPE
    x0, x1 = x[..., lo:lo + D_ROPE:2], x[..., lo + 1:lo + D_ROPE:2]
    y = x.clone()
    y[..., lo:lo + D_ROPE:2] = x0 * c - x1 * s
    y[..., lo + 1:lo + D_ROPE:2] = x0 * s + x1 * c
    return y


def dequant_out(out_fp8, out_sf, s_q):
    """(out_fp8 [s_q, N_WV, 8*D] e4m3, out_sf packed ue8m0, 1 byte per 32 elements) -> O [s_q, H, D] fp64"""
    x = out_fp8.view(s_q, -1).double()
    sfi = out_sf.view(s_q, -1).to(torch.int64)
    b = torch.stack([(sfi >> (8 * k)) & 255 for k in range(4)], -1).reshape(s_q, -1)
    x = x.view(s_q, -1, 32) * torch.exp2(b.double() - 127)[..., None]
    wv = H_Q // N_WV
    return x.view(s_q, N_WV, D // 32, wv, 32).transpose(2, 3).reshape(s_q, H_Q, D)


def permute_q(q):
    """plain [s_q, H, D] -> the permuted layout of permute_q_b_proj: element (c, h, i) = q[h, c*16 + i]"""
    s_q = q.shape[0]
    return q.view(s_q, H_Q, D // 16, 16).transpose(1, 2).reshape(s_q, H_Q, D).contiguous()


def run(q, kv, idx, sink, sm_scale, topk_length=None, rope=None, enable_q_norm=False):
    """q: plain [s_q, H, D] bf16; rope: (token_positions, cos_sin_cache), None for the identity"""
    s_q = q.shape[0]
    pos, cs = identity_rope(s_q) if rope is None else rope
    out_fp8, out_sf, max_logits, lse = flash_mla.fused_norm_rope_attn_rope_cast.prefill(
        enable_q_norm, RMS_NORM_EPS, pos.int().cuda(), False, D_ROPE, cs.float().cuda(), N_WV, 32, True, True, True,
        permute_q(q).cuda(), kv.cuda().view(kv.shape[0], 1, D), idx.cuda().view(s_q, 1, -1), sm_scale, D,
        sink.float().cuda(),
        torch.full((s_q,), idx.shape[1], dtype=torch.int32).cuda() if topk_length is None else topk_length.int().cuda())
    torch.cuda.synchronize()
    return dequant_out(out_fp8, out_sf, s_q), lse.double(), max_logits


def metrics(o, lse, ro, rl):
    d = (o - ro).flatten(); r = ro.flatten()
    fin = torch.isfinite(lse) & torch.isfinite(rl)
    le = (lse - rl)[fin].abs()
    tok_cos = torch.nn.functional.cosine_similarity(o.reshape(o.shape[0], -1), ro.reshape(o.shape[0], -1), dim=1)
    return {"rel_l2": (d.norm() / r.norm()).item(),
            "tok_cos": tok_cos.min().item(),
            "lse_mean": le.mean().item() if le.numel() else float("nan"),
            "nonfinite": int((~torch.isfinite(o)).sum().item())}


def passes(m, limits):
    return (m["nonfinite"] == 0 and m["rel_l2"] <= limits["rel_l2"] and m["tok_cos"] >= limits["tok_cos"]
            and m["lse_mean"] <= limits["lse_mean"])


def to_e4m3(ws):
    return ws.float().clamp(-448, 448).to(torch.float8_e4m3fn)


@requires_sm100
@pytest.mark.parametrize("topk", [128, 640])
@pytest.mark.parametrize("dist", ["bench", "small", "unselected"])
def test_fp8_prefill_precision(dist, topk):
    """vs fp64 on the true kv, identity RoPE, enable_q_norm False. Inputs outside the contract (|q| or |kv| beyond
    e4m3's normal range: |x| > 448 saturates, |x| < 2^-6 is subnormal) are not tested: Q and KV enter the tensor cores
    as e4m3 with unit scale, so the caller must keep them in range"""
    s_q, s_kv = 1024, 8192
    q, ws, idx, sink = make_case(dist, s_q, s_kv, topk, seed=1234 + topk)
    sm_scale = D ** -0.5
    ro, rl = reference(q.float(), ws.float(), idx, sink, sm_scale)

    # the bf16 op of this tree on the same workspace: the reference / output decoding sanity check
    o, lse, _ = run(q, ws, idx, sink, sm_scale)
    m = metrics(o, lse, ro, rl)
    assert passes(m, FP64_LIMITS["bf16"]), f"bf16 {dist}: {m}"

    o, lse, _ = run(q, to_e4m3(ws), idx, sink, sm_scale)
    m = metrics(o, lse, ro, rl)
    assert passes(m, FP64_LIMITS[f"fp8 {dist}"]), f"fp8 {dist}: {m}"


# Limits vs the bf16 op on the random-RoPE case, same rule as above. The scores are sharp, so Q's e4m3 rounding shows
# more than in `bench`. Simulated bugs (the mutant references below and in ACCURACY.md) land at: RoPE on a wrong Q tile
# rel_l2 >= 0.98, O without its inverse RoPE 0.50, round-toward-zero instead of RN for the e4m3 cast of kv (or of Q)
# lse 0.21-0.27
ROPE_LIMITS = {"rel_l2": 0.12, "tok_cos": 0.984, "lse_mean": 0.055}   # 0.0917, 0.98886, 0.040


@requires_sm100
@pytest.mark.parametrize("topk", [128, 640])
def test_fp8_prefill_rope(topk):
    """vs the bf16 op (same q, kv workspace, token_positions, cos_sin_cache): a random cos_sin_cache and random
    positions, so a wrong mapping of the fp8 kernel's Q tiles to rails (RoPE is applied by tile index) or a wrong
    inverse RoPE of O's last 64 dims fails. The last part checks that this case can see such bugs: references with one
    built in must fail the limits"""
    s_q, s_kv = 1024, 8192
    q, ws, idx, sink, rope, sm_scale = make_rope_case(s_q, s_kv, topk, seed=4321 + topk)
    assert q.float().abs().max().item() <= 448
    ro, rl, _ = run(q, ws, idx, sink, sm_scale, rope=rope)
    o, lse, _ = run(q, to_e4m3(ws), idx, sink, sm_scale, rope=rope)
    m = metrics(o, lse, ro, rl)
    assert passes(m, ROPE_LIMITS), f"fp8 vs bf16: {m}"

    # Q: undo the op's RoPE of the last tile and rotate tile 6 (the other half of the same 128-dim e4m3 swizzle row)
    q_wrong_tile = rope_tile(rope_tile(q.float(), rope, D // D_ROPE - 1, inverse=True), rope, D // D_ROPE - 2)
    mutants = {"Q RoPE on dims 384-447": run(q_wrong_tile.bfloat16(), ws, idx, sink, sm_scale, rope=rope)[:2],
               "O without inverse RoPE": (rope_tile(ro, rope, D // D_ROPE - 1), rl)}
    for name, (mo, ml) in mutants.items():
        mm = metrics(o, lse, mo, ml)
        assert not passes(mm, ROPE_LIMITS), f"insensitive to '{name}': {mm}"


@requires_sm100
def test_fp8_prefill_rejects_q_norm():
    """The fp8 kv supports enable_q_norm False only; True is rejected (the bf16 kv keeps both)"""
    q, ws, idx, sink = make_case("bench", 128, 1024, 128, seed=5)
    with pytest.raises(RuntimeError, match="enable_q_norm"):
        run(q, to_e4m3(ws), idx, sink, D ** -0.5, enable_q_norm=True)


@requires_sm100
def test_fp8_prefill_p_offset_keeps_lse_sink_and_max_logits():
    """The fp8 instances scale P by 4 before its e4m3 cast and compensate in the max they publish to the epilogue.
    Check the three places that use the absolute softmax sum: lse, the sink weighting of O (a sink at the row's lse
    gives it ~half the weight, so a factor-4 slip would move O by 60-75%), and max_logits; plus rows without any valid
    token (lse +inf, max_logits -inf, O zero)"""
    s_q, s_kv, topk = 512, 4096, 640
    q, ws, idx, _ = make_case("bench", s_q, s_kv, topk, seed=21)
    sm_scale = D ** -0.5
    _, rl0 = reference(q.float(), ws.float(), idx, torch.full((H_Q,), -float("inf")), sm_scale)
    sink = rl0.mean(0).float().cpu()                   # per head, at the typical lse: the sink takes ~half the weight
    ro, rl = reference(q.float(), ws.float(), idx, sink, sm_scale)
    K = ws.float().cuda()[idx.long().cuda()]
    rml = (torch.einsum("shd,std->sht", q.float().cuda(), K) * sm_scale).amax(-1).double()
    tl = torch.full((s_q,), topk, dtype=torch.int32); tl[::7] = 0      # every 7th row attends to nothing
    o, lse, ml = run(q, to_e4m3(ws), idx, sink, sm_scale, topk_length=tl)
    v = (tl > 0).cuda()
    m = metrics(o[v], lse[v], ro[v], rl[v])
    assert passes(m, FP64_LIMITS["fp8 bench"]), f"fp8 with a sink at the lse: {m}"
    assert (ml.double()[v] - rml[v]).abs().mean().item() < MAX_LOGITS_MEAN_LIMIT
    e = ~v
    assert torch.isinf(lse[e]).all() and (lse[e] > 0).all()
    assert torch.isinf(ml[e]).all() and (ml[e] < 0).all()
    assert (o[e] == 0).all()


# Limits vs fp64 on the partially valid case, on the rows with a valid entry (same rule as above; measured worst case
# over topk 128 / 640 below). References with the validity rule broken land at rel_l2 >= 0.39 (counting the
# entry at topk_length), 0.44 (dropping the partial last block), 0.78 (reading invalid indices as index mod s_kv), 1.9
# (ignoring topk_length), all with min per-token cos <= 0.69. Measured (rel_l2, tok_cos, lse mean, max_logits mean):
# bf16 0.0265, 0.99957, 2.5e-7, 5.1e-9; fp8 0.0416, 0.99889, 6.1e-5, 3.0e-4
PARTIAL_LIMITS = {
    "bf16": {"rel_l2": 0.035, "tok_cos": 0.9994, "lse_mean": 3.5e-7, "max_logits_mean": 7.5e-9},
    "fp8": {"rel_l2": 0.055, "tok_cos": 0.9984, "lse_mean": 8.5e-5, "max_logits_mean": 4.5e-4},
}


@requires_sm100
@pytest.mark.parametrize("topk", [128, 640])
def test_fp8_prefill_partially_valid_blocks(topk):
    """Blocks of 64 indices with only some entries valid, so the fp8 softmax takes its masked path (the all-valid fast
    path skips the per-element selects): entry j of row i counts iff j < topk_length[i] and 0 <= indices[i, j] < s_kv
    (negative or >= s_kv indices are invalid; the entries from topk_length on are ignored whatever they hold). Checked
    against an fp64 reference with that rule, the bf16 op on the same inputs, and references with the rule broken,
    which must fail the limits"""
    s_q, s_kv = 1024, 8192
    q, ws, idx, sink, tl = make_partial_case(s_q, s_kv, topk, seed=777 + topk)
    sm_scale = D ** -0.5
    ro, rl, rml = reference_masked(q.float(), ws.float(), idx, sink, sm_scale, tl)
    v = torch.isfinite(rl).all(-1)                     # rows with at least one valid entry
    e = ~v
    assert 0 < int(e.sum()) < s_q // 4
    for op, kv in (("bf16", ws), ("fp8", to_e4m3(ws))):
        o, lse, ml = run(q, kv, idx, sink, sm_scale, topk_length=tl)
        m = metrics(o[v], lse[v], ro[v], rl[v])
        assert passes(m, PARTIAL_LIMITS[op]), f"{op}: {m}"
        assert (ml.double()[v] - rml[v]).abs().mean().item() <= PARTIAL_LIMITS[op]["max_logits_mean"], op
        assert torch.isinf(lse[e]).all() and (lse[e] > 0).all(), op
        assert torch.isinf(ml[e]).all() and (ml[e] < 0).all(), op
        assert (o[e] == 0).all(), op

    # o, lse: the fp8 op. References with the validity rule broken, on the rows that still have a valid entry
    full = torch.full_like(tl, topk)
    mutants = {"topk_length ignored": (idx, full),
               "partial last block dropped": (idx, tl // 64 * 64),
               "topk_length off by one (j <= topk_length counts)": (idx, torch.clamp(tl + 1, max=topk)),
               "invalid indices read as index mod s_kv": (idx.remainder(s_kv), tl)}
    for name, (mi, mt) in mutants.items():
        mo, ml_lse, _ = reference_masked(q.float(), ws.float(), mi, sink, sm_scale, mt)
        w = v & torch.isfinite(ml_lse).all(-1)
        mm = metrics(o[w], lse[w], mo[w], ml_lse[w])
        assert not passes(mm, PARTIAL_LIMITS["fp8"]), f"insensitive to '{name}': {mm}"


@requires_sm100
def test_fp8_prefill_deterministic():
    q, ws, idx, sink, rope, sm_scale = make_rope_case(512, 4096, 640, seed=11)
    kv8 = to_e4m3(ws)
    r0 = run(q, kv8, idx, sink, sm_scale, rope=rope)
    for _ in range(5):
        r1 = run(q, kv8, idx, sink, sm_scale, rope=rope)
        assert all(torch.equal(a, b) for a, b in zip(r0, r1))
