"""Tests of the fp8 sparse prefill of fused_norm_rope_attn_rope_cast.prefill (a float8_e4m3fn `kv`, h_q = 64).

The reference is fp64 attention on the TRUE kv values (the bf16 workspace a caller quantizes from), not on the e4m3
tensor handed to the kernel, so range losses of the fp8 kv (saturation above 448, subnormals) show up in the error.
The bf16 op of the same tree (`kv` in bfloat16) runs on the same inputs as a second reference. RoPE is the identity
(cos = 1, sin = 0): it is not what this test is about, and it keeps Q / O exact; enable_q_norm is False.

The fp8 kv is the bf16 workspace cast to e4m3 directly (unit scale; the kernel also casts Q with unit scale and
scales P by 4 before its e4m3 cast). Distributions: `bench` (randn * 0.1), `small` (randn * 0.01, mostly e4m3
subnormals), `unselected` (a few outlier tokens in the kv that no query selects), and, outside the contract
(|q|, |kv| within e4m3's normal range), `outlier` (4 KV channels x 300, |kv| up to ~1.3k), `q_big` / `q_small`
(|q| ~ 20 x 448 / ~1e-4, sharp attention so the result depends on Q's precision).

Needs an SM100 GPU and the compiled flash_mla.  python -m pytest tests/test_fp8_prefill.py
"""
import pytest
import torch

import flash_mla

H_Q, D, D_ROPE, N_WV = 64, 512, 64, 8

# Limits vs the fp64 reference on the true values. e4m3 Q/K/V/P plus the per-32 fp8 output cast (which the bf16 op has
# too) cost an O rel_l2 of ~4-5e-2 on in-range data and ~6e-2 on a mostly subnormal kv (`small`)
LIMITS = {"rel_l2": 0.10, "cos": 0.995, "lse_mean": 0.5}

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the fused prefill needs an SM100f GPU",
)


def make_case(dist, s_q, s_kv, topk, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q_mul = {"q_big": 20.0 * 448 / 0.4, "q_small": 1e-3}.get(dist, 1.0)
    q = (torch.randn(s_q, H_Q, D, generator=g) * 0.1 * q_mul).to(torch.bfloat16)
    sm_scale_mul = (30.0 if dist in ("q_big", "q_small") else 1.0) / q_mul
    std = {"bench": 0.1, "outlier": 1.0, "small": 0.01, "q_big": 1.0, "q_small": 1.0, "unselected": 0.1}[dist]
    kv = torch.randn(s_kv, D, generator=g) * std
    n_sel = s_kv
    if dist == "outlier":
        ch = torch.randperm(D, generator=g)[:4]
        kv[:, ch] *= 300.0
    if dist == "unselected":
        n_sel = s_kv - 16
        kv[n_sel:] *= 1e4     # amax ~ 4e3: outside every query's selection
    ws = kv.to(torch.bfloat16)
    idx = torch.stack([torch.randperm(n_sel, generator=g)[:topk] for _ in range(s_q)]).to(torch.int32)
    sink = torch.randn(H_Q, generator=g)
    return q, ws, idx, sink, sm_scale_mul


def reference(q, kv_true, idx, sink, sm_scale):
    """fp64 sparse attention with sink: O [s_q, H, D], lse [s_q, H] (natural log, without the sink)"""
    q = q.double().cuda(); kv = kv_true.double().cuda(); idx = idx.long().cuda(); sink = sink.double().cuda()
    K = kv[idx]
    lg = torch.einsum("shd,std->sht", q, K) * sm_scale
    lse = torch.logsumexp(lg, -1)
    den = torch.logaddexp(lse, sink[None, :])
    return torch.einsum("sht,std->shd", torch.exp(lg - den[..., None]), K), lse


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


def run(q, kv, idx, sink, sm_scale, topk_length=None):
    s_q = q.shape[0]
    cs = torch.cat([torch.ones(1, D_ROPE // 2), torch.zeros(1, D_ROPE // 2)], 1).float().cuda()   # identity RoPE
    pos = torch.zeros(s_q, dtype=torch.int32, device="cuda")
    out_fp8, out_sf, max_logits, lse = flash_mla.fused_norm_rope_attn_rope_cast.prefill(
        False, 1e-6, pos, False, D_ROPE, cs, N_WV, 32, True, True, True,
        permute_q(q).cuda(), kv.cuda().view(kv.shape[0], 1, D), idx.cuda().view(s_q, 1, -1), sm_scale, D,
        sink.float().cuda(),
        torch.full((s_q,), idx.shape[1], dtype=torch.int32).cuda() if topk_length is None else topk_length.int().cuda())
    torch.cuda.synchronize()
    return dequant_out(out_fp8, out_sf, s_q), lse.double(), max_logits


def metrics(o, lse, ro, rl):
    d = (o - ro).flatten(); r = ro.flatten()
    fin = torch.isfinite(lse) & torch.isfinite(rl)
    le = (lse - rl)[fin].abs()
    return {"rel_l2": (d.norm() / r.norm()).item(),
            "cos": torch.nn.functional.cosine_similarity(o.flatten(), r, dim=0).item(),
            "lse_mean": le.mean().item() if le.numel() else float("nan"),
            "nonfinite": int((~torch.isfinite(o)).sum().item())}


def check(m, expect_pass, what):
    ok = (m["nonfinite"] == 0 and m["rel_l2"] <= LIMITS["rel_l2"] and m["cos"] >= LIMITS["cos"]
          and m["lse_mean"] <= LIMITS["lse_mean"])
    assert ok == expect_pass, f"{what}: expected {'pass' if expect_pass else 'fail'}, got {m}"


def to_e4m3(ws):
    return ws.float().clamp(-448, 448).to(torch.float8_e4m3fn)


OUT_OF_RANGE = pytest.mark.xfail(strict=True, reason=(
    "outside the fp8 kv / q contract, not a kernel bug: Q and KV enter the tensor cores as e4m3 with unit scale, so "
    "the "
    "caller must keep |kv| and |q| within e4m3's normal range (|x| <= 448 saturates beyond, |x| < 2^-6 is subnormal)"))
CASES = ["bench", "small", "unselected", pytest.param("outlier", marks=OUT_OF_RANGE),
         pytest.param("q_big", marks=OUT_OF_RANGE), pytest.param("q_small", marks=OUT_OF_RANGE)]


@requires_sm100
@pytest.mark.parametrize("topk", [128, 640])
@pytest.mark.parametrize("dist", CASES)
def test_fp8_prefill_precision(dist, topk):
    s_q, s_kv = 1024, 8192
    q, ws, idx, sink, sm_mul = make_case(dist, s_q, s_kv, topk, seed=1234 + topk)
    sm_scale = D ** -0.5 * sm_mul
    ro, rl = reference(q.float(), ws.float(), idx, sink, sm_scale)

    # the bf16 op of this tree on the same workspace: the reference / output decoding sanity check
    o, lse, _ = run(q, ws, idx, sink, sm_scale)
    check(metrics(o, lse, ro, rl), True, f"bf16 {dist}")

    o, lse, _ = run(q, to_e4m3(ws), idx, sink, sm_scale)
    check(metrics(o, lse, ro, rl), True, f"fp8 {dist}")


@requires_sm100
def test_fp8_prefill_p_offset_keeps_lse_sink_and_max_logits():
    """The fp8 instances scale P by 4 before its e4m3 cast and compensate in the max they publish to the epilogue.
    Check the three places that use the absolute softmax sum: lse, the sink weighting of O (a sink at the row's lse
    gives it ~half the weight, so a factor-4 slip would move O by 60-75%), and max_logits; plus rows without any valid
    token (lse +inf, max_logits -inf, O zero)"""
    s_q, s_kv, topk = 512, 4096, 640
    q, ws, idx, _, _ = make_case("bench", s_q, s_kv, topk, seed=21)
    sm_scale = D ** -0.5
    _, rl0 = reference(q.float(), ws.float(), idx, torch.full((H_Q,), -float("inf")), sm_scale)
    sink = rl0.mean(0).float().cpu()                   # per head, at the typical lse: the sink takes ~half the weight
    ro, rl = reference(q.float(), ws.float(), idx, sink, sm_scale)
    K = ws.float().cuda()[idx.long().cuda()]
    rml = (torch.einsum("shd,std->sht", q.float().cuda(), K) * sm_scale).amax(-1).double()
    tl = torch.full((s_q,), topk, dtype=torch.int32); tl[::7] = 0      # every 7th row attends to nothing
    o, lse, ml = run(q, to_e4m3(ws), idx, sink, sm_scale, topk_length=tl)
    v = (tl > 0).cuda()
    check(metrics(o[v], lse[v], ro[v], rl[v]), True, "fp8 with a sink at the lse")
    assert (ml.double()[v] - rml[v]).abs().mean().item() < LIMITS["lse_mean"]
    e = ~v
    assert torch.isinf(lse[e]).all() and (lse[e] > 0).all()
    assert torch.isinf(ml[e]).all() and (ml[e] < 0).all()
    assert (o[e] == 0).all()


@requires_sm100
def test_fp8_prefill_deterministic():
    q, ws, idx, sink, _ = make_case("bench", 512, 4096, 640, seed=11)
    kv8 = to_e4m3(ws)
    r0 = run(q, kv8, idx, sink, D ** -0.5)
    for _ in range(5):
        r1 = run(q, kv8, idx, sink, D ** -0.5)
        assert all(torch.equal(a, b) for a, b in zip(r0, r1))
