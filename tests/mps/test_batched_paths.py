"""Phase 3a regression tests: the batched kv-cache decode path and the batched
varlen driver, specifically their ragged/skewed corners.

The batched formulations pad to the batch max and mask; a leaky mask is a
silent-wrongness bug, so these tests pin:

- wildly varying ``cache_seqlens`` (including 0) with in-place append, GQA,
  causal, ``cache_leftpad``/``cache_batch_idx``, windows, ALiBi — against the
  per-sequence CPU oracle;
- a single-token decode step (the serving hot path);
- batched-vs-looped varlen parity (both strategies forced) across causal /
  window / GQA / seqused / zero-length sequences / unequal q-k lengths,
  forward AND backward;
- fully-masked rows through the masked-SDPA no-lse path: exact zeros, exact
  zero gradients, never NaN;
- the skew heuristic: extreme length skew must pick the loop, launch-bound
  shapes must pick the batched call.
"""

import pytest
import torch

from checks import check_tensor_budget
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from flash_attn.cute.testing import attention_ref
from flash_attn.mps.varlen import _prefer_batched, mps_flash_attn_varlen

if not torch.backends.mps.is_available():
    pytest.skip("MPS not available", allow_module_level=True)

DEVICE = "mps"


def _ref_pair(q, k, v, **kwargs):
    """(out_ref fp32-upcast, out_pt in-dtype) on CPU, attention_ref conventions."""
    out_ref, _ = attention_ref(q, k, v, None, None, **kwargs)
    out_pt, _ = attention_ref(
        q, k, v, None, None, upcast=False, reorder_ops=True, **kwargs
    )
    return out_ref, out_pt


# ---------------------------------------------------------------------------
# kv-cache decode (fa2_backend.fwd_kvcache, batched formulation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_kvcache_ragged_cache_seqlens_append_gqa(dtype):
    """Wildly varying cache_seqlens (incl. 0) + in-place append + GQA + causal."""
    torch.manual_seed(100)
    batch, s_cache, s_new, h, h_kv, d = 8, 512, 4, 8, 2, 64
    lens = torch.tensor(
        [0, 1, 3, 17, 100, 300, 451, 508], dtype=torch.int32, device=DEVICE
    )
    k_cache = torch.randn(batch, s_cache, h_kv, d, device=DEVICE).to(dtype)
    v_cache = torch.randn(batch, s_cache, h_kv, d, device=DEVICE).to(dtype)
    k_cache_orig, v_cache_orig = k_cache.clone(), v_cache.clone()
    q = torch.randn(batch, s_new, h, d, device=DEVICE).to(dtype)
    k_new = torch.randn(batch, s_new, h_kv, d, device=DEVICE).to(dtype)
    v_new = torch.randn(batch, s_new, h_kv, d, device=DEVICE).to(dtype)
    out, lse = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k_new,
        v_new,
        cache_seqlens=lens,
        causal=True,
        return_softmax_lse=True,
    )
    assert not torch.isnan(out).any() and not torch.isnan(lse).any()
    for i in range(batch):
        L = int(lens[i])
        # in-place append at the right offset, rest of the cache untouched
        assert (k_cache[i, L : L + s_new] == k_new[i]).all()
        assert (v_cache[i, L : L + s_new] == v_new[i]).all()
        assert (k_cache[i, :L] == k_cache_orig[i, :L]).all()
        assert (k_cache[i, L + s_new :] == k_cache_orig[i, L + s_new :]).all()
        kf = torch.cat([k_cache_orig[i, :L], k_new[i]], 0).unsqueeze(0).cpu()
        vf = torch.cat([v_cache_orig[i, :L], v_new[i]], 0).unsqueeze(0).cpu()
        out_ref, out_pt = _ref_pair(q[i : i + 1].cpu(), kf, vf, causal=True)
        check_tensor_budget(
            "out",
            out[i : i + 1],
            out_ref,
            out_pt,
            dtype,
            detail=f"kvcache ragged b{i} L={L}",
        )


def test_kvcache_single_token_decode_step():
    """The serving hot path: sq=1, ragged lens, append one token per sequence."""
    torch.manual_seed(101)
    dtype = torch.float16
    batch, s_cache, h, h_kv, d = 32, 256, 8, 2, 128
    lens = torch.randint(0, s_cache - 1, (batch,), dtype=torch.int32, device=DEVICE)
    lens[0], lens[1] = 0, s_cache - 1  # extremes
    k_cache = torch.randn(batch, s_cache, h_kv, d, device=DEVICE).to(dtype)
    v_cache = torch.randn(batch, s_cache, h_kv, d, device=DEVICE).to(dtype)
    k_cache_orig, v_cache_orig = k_cache.clone(), v_cache.clone()
    q = torch.randn(batch, 1, h, d, device=DEVICE).to(dtype)
    k_new = torch.randn(batch, 1, h_kv, d, device=DEVICE).to(dtype)
    v_new = torch.randn(batch, 1, h_kv, d, device=DEVICE).to(dtype)
    out = flash_attn_with_kvcache(
        q, k_cache, v_cache, k_new, v_new, cache_seqlens=lens, causal=True
    )
    assert not torch.isnan(out).any()
    for i in range(batch):
        L = int(lens[i])
        kf = torch.cat([k_cache_orig[i, :L], k_new[i]], 0).unsqueeze(0).cpu()
        vf = torch.cat([v_cache_orig[i, :L], v_new[i]], 0).unsqueeze(0).cpu()
        out_ref, out_pt = _ref_pair(q[i : i + 1].cpu(), kf, vf, causal=True)
        check_tensor_budget(
            "out", out[i : i + 1], out_ref, out_pt, dtype, detail=f"decode1 b{i} L={L}"
        )


def test_kvcache_ragged_leftpad_batch_idx_window():
    """cache_leftpad + cache_batch_idx + sliding window over ragged lengths."""
    torch.manual_seed(102)
    dtype = torch.float16
    batch, s_cache, sq, h, d = 4, 96, 3, 4, 64
    k_cache = torch.randn(batch, s_cache, h, d, device=DEVICE).to(dtype)
    v_cache = torch.randn(batch, s_cache, h, d, device=DEVICE).to(dtype)
    q = torch.randn(batch, sq, h, d, device=DEVICE).to(dtype)
    lens = torch.tensor([90, 7, 33, 64], dtype=torch.int32, device=DEVICE)
    leftpad = torch.tensor([5, 0, 30, 2], dtype=torch.int32, device=DEVICE)
    batch_idx = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=DEVICE)
    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=lens,
        cache_leftpad=leftpad,
        cache_batch_idx=batch_idx,
        causal=True,
        window_size=(16, -1),
    )
    assert not torch.isnan(out).any()
    for i in range(batch):
        bi, L, lp = int(batch_idx[i]), int(lens[i]), int(leftpad[i])
        kf = k_cache[bi, lp:L].unsqueeze(0).cpu()
        vf = v_cache[bi, lp:L].unsqueeze(0).cpu()
        out_ref, out_pt = _ref_pair(
            q[i : i + 1].cpu(), kf, vf, causal=True, window_size=(16, None)
        )
        check_tensor_budget(
            "out",
            out[i : i + 1],
            out_ref,
            out_pt,
            dtype,
            detail=f"kvcache lp/bidx/win b{i}",
        )


def test_kvcache_ragged_alibi():
    """ALiBi bias must use each sequence's *effective* cache length for its
    diagonal, not the padded batch max."""
    torch.manual_seed(103)
    dtype = torch.float16
    batch, s_cache, sq, h, d = 4, 80, 2, 4, 64
    k_cache = torch.randn(batch, s_cache, h, d, device=DEVICE).to(dtype)
    v_cache = torch.randn(batch, s_cache, h, d, device=DEVICE).to(dtype)
    q = torch.randn(batch, sq, h, d, device=DEVICE).to(dtype)
    lens = torch.tensor([80, 2, 37, 11], dtype=torch.int32, device=DEVICE)
    slopes = torch.rand(h, device=DEVICE, dtype=torch.float32) * 0.3
    out = flash_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens=lens, causal=True, alibi_slopes=slopes
    )
    assert not torch.isnan(out).any()
    for i in range(batch):
        L = int(lens[i])
        kf = k_cache[i, :L].unsqueeze(0).cpu()
        vf = v_cache[i, :L].unsqueeze(0).cpu()
        row = torch.arange(sq).view(-1, 1)
        col = torch.arange(L)
        rel = (row + (L - sq) - col).abs().float()  # per-sequence diagonal
        bias = (-slopes.cpu().view(1, h, 1, 1) * rel).to(torch.float32)
        out_ref, out_pt = _ref_pair(
            q[i : i + 1].cpu(), kf, vf, causal=True, attn_bias=bias
        )
        check_tensor_budget(
            "out",
            out[i : i + 1],
            out_ref,
            out_pt,
            dtype,
            detail=f"kvcache alibi b{i} L={L}",
        )


def test_kvcache_no_valid_keys_rows():
    """Batch elements with cache_seqlens == 0 and nothing appended: out exactly
    0, lse +inf (FA2 sign), no NaN — while their neighbors compute normally."""
    torch.manual_seed(104)
    dtype = torch.float16
    batch, s_cache, sq, h, d = 3, 64, 2, 4, 64
    k_cache = torch.randn(batch, s_cache, h, d, device=DEVICE).to(dtype)
    v_cache = torch.randn(batch, s_cache, h, d, device=DEVICE).to(dtype)
    q = torch.randn(batch, sq, h, d, device=DEVICE).to(dtype)
    lens = torch.tensor([0, 40, 0], dtype=torch.int32, device=DEVICE)
    out, lse = flash_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens=lens, causal=True, return_softmax_lse=True
    )
    assert not torch.isnan(out).any()
    for i in (0, 2):
        assert (out[i] == 0).all(), "no-keys rows must be exactly zero"
        assert (lse[i] == float("inf")).all(), "no-keys rows must report +inf lse"
    assert torch.isfinite(lse[1]).all()
    out_ref, out_pt = _ref_pair(
        q[1:2].cpu(),
        k_cache[1, :40].unsqueeze(0).cpu(),
        v_cache[1, :40].unsqueeze(0).cpu(),
        causal=True,
    )
    check_tensor_budget(
        "out", out[1:2], out_ref, out_pt, dtype, detail="kvcache zero-len b1"
    )


# ---------------------------------------------------------------------------
# varlen: batched vs looped strategy parity (both forced)
# ---------------------------------------------------------------------------


def _packed(lens_q, lens_k, h, h_kv, d, dtype, seed, requires_grad=False):
    torch.manual_seed(seed)
    cu_q = torch.tensor([0] + list(torch.tensor(lens_q).cumsum(0)), dtype=torch.int32)
    cu_k = torch.tensor([0] + list(torch.tensor(lens_k).cumsum(0)), dtype=torch.int32)
    q = torch.randn(int(cu_q[-1]), h, d, device=DEVICE).to(dtype)
    k = torch.randn(int(cu_k[-1]), h_kv, d, device=DEVICE).to(dtype)
    v = torch.randn(int(cu_k[-1]), h_kv, d, device=DEVICE).to(dtype)
    if requires_grad:
        for t in (q, k, v):
            t.requires_grad_()
    return q, k, v, cu_q.to(DEVICE), cu_k.to(DEVICE)


def _assert_same(a, b, dtype, what, atol=None):
    if atol is None:
        atol = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}[dtype]
    assert not torch.isnan(a).any(), f"{what}: NaN (batched)"
    assert not torch.isnan(b).any(), f"{what}: NaN (looped)"
    inf_a, inf_b = torch.isinf(a), torch.isinf(b)
    assert torch.equal(inf_a, inf_b), f"{what}: inf positions differ between strategies"
    if inf_a.any():
        assert torch.equal(a[inf_a], b[inf_b]), f"{what}: inf signs differ"
    diff = (
        (a[~inf_a].float() - b[~inf_b].float()).abs().max().item()
        if (~inf_a).any()
        else 0.0
    )
    assert diff <= atol, f"{what}: batched vs looped max diff {diff:.3e} > {atol}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize(
    "causal,window,gqa,seqused_k",
    [
        (True, (None, None), False, False),
        (False, (None, None), True, True),
        (True, (5, None), True, False),
        (False, (7, 3), False, True),
    ],
    ids=["causal", "gqa-seqused", "causal-window-gqa", "window-lr-seqused"],
)
def test_varlen_batched_matches_looped(dtype, causal, window, gqa, seqused_k):
    """Ragged lengths incl. a zero-length sequence: both strategies must agree
    (out, lse, and grads) — the loop is the Phase-1 parity-pinned reference."""
    lens = [37, 0, 128, 5, 64, 1, 96, 33]
    h, h_kv = 8, (2 if gqa else 8)
    q, k, v, cu_q, cu_k = _packed(
        lens, lens, h, h_kv, 64, dtype, seed=200, requires_grad=True
    )
    su_k = (
        torch.tensor(
            [min(n, max(n - 3, 0)) for n in lens], dtype=torch.int32, device=DEVICE
        )
        if seqused_k
        else None
    )
    grads = {}
    outs, lses = {}, {}
    dout = torch.randn(q.shape, device=DEVICE, dtype=dtype)
    for forced, name in ((True, "batched"), (False, "looped")):
        q.grad = k.grad = v.grad = None
        out, lse = mps_flash_attn_varlen(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            seqused_k=su_k,
            causal=causal,
            window_size=window,
            return_lse=True,
            _force_batched=forced,
        )
        out.backward(dout)
        outs[name], lses[name] = out.detach(), lse.detach()
        grads[name] = (q.grad.clone(), k.grad.clone(), v.grad.clone())
    _assert_same(outs["batched"], outs["looped"], dtype, "out")
    _assert_same(lses["batched"], lses["looped"], torch.float32, "lse", atol=1e-4)
    for g_b, g_l, what in zip(grads["batched"], grads["looped"], ("dq", "dk", "dv")):
        _assert_same(g_b, g_l, dtype, what)


def test_varlen_batched_unequal_qk_fully_masked_rows():
    """causal with seqlen_q > seqlen_k per sequence: leading rows have no
    visible keys. Batched path (both lse and masked-SDPA no-lse) must produce
    exact zeros / -inf lse / exact-zero grads there — never NaN."""
    dtype = torch.float16
    lens_q, lens_k = [48, 16, 7], [16, 16, 2]
    q, k, v, cu_q, cu_k = _packed(
        lens_q, lens_k, 4, 4, 64, dtype, seed=201, requires_grad=True
    )
    out, lse = mps_flash_attn_varlen(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        causal=True,
        return_lse=True,
        _force_batched=True,
    )
    assert not torch.isnan(out).any() and not torch.isnan(lse).any()
    # per-sequence: first (sq_i - sk_i) rows are fully masked
    off = 0
    for sq_i, sk_i in zip(lens_q, lens_k):
        n_masked = max(sq_i - sk_i, 0)
        assert (out[off : off + n_masked] == 0).all()
        assert (lse[:, off : off + n_masked] == float("-inf")).all()
        assert torch.isfinite(lse[:, off + n_masked : off + sq_i]).all()
        off += sq_i
    # no-lse (masked SDPA) path, with gradients
    q.grad = k.grad = v.grad = None
    out2, _ = mps_flash_attn_varlen(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        causal=True,
        return_lse=False,
        _force_batched=True,
    )
    assert not torch.isnan(out2).any()
    assert (out2[:32] == 0).all(), "SDPA path: fully-masked rows must be exactly 0"
    out2.backward(torch.randn_like(out2))
    for g, what in ((q.grad, "dq"), (k.grad, "dk"), (v.grad, "dv")):
        assert not torch.isnan(g).any(), f"{what}: NaN gradient"
    assert (q.grad[:32] == 0).all(), "fully-masked rows must have exactly-zero dq"


def test_varlen_empty_and_single_token_sequences():
    """cu_seqlens with zero-length and length-1 sequences, both strategies,
    forward and backward, against the CPU oracle."""
    dtype = torch.float16
    lens = [0, 1, 0, 23, 1, 0]
    for forced in (True, False):
        q, k, v, cu_q, cu_k = _packed(
            lens, lens, 4, 4, 64, dtype, seed=202, requires_grad=True
        )
        out, lse = mps_flash_attn_varlen(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            causal=True,
            return_lse=True,
            _force_batched=forced,
        )
        assert not torch.isnan(out).any()
        assert torch.isfinite(lse).all()
        out.sum().backward()
        assert not torch.isnan(q.grad).any()
        off = 0
        for n in lens:
            if n == 0:
                continue
            qi = q[off : off + n].detach().cpu().unsqueeze(0)
            ki = k[off : off + n].detach().cpu().unsqueeze(0)
            vi = v[off : off + n].detach().cpu().unsqueeze(0)
            out_ref, out_pt = _ref_pair(qi, ki, vi, causal=True)
            check_tensor_budget(
                "out",
                out[off : off + n].unsqueeze(0),
                out_ref,
                out_pt,
                dtype,
                detail=f"varlen tiny n={n} forced={forced}",
            )
            off += n


def test_varlen_extreme_skew_public_api():
    """One 2048-token sequence among 31 tiny ones, through the real FA2 seam
    (which auto-picks a strategy): parity per sequence with the oracle."""
    dtype = torch.float16
    lens = [2048] + [32] * 31
    q, k, v, cu_q, cu_k = _packed(lens, lens, 4, 4, 64, dtype, seed=203)
    out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max(lens), max(lens), causal=True)
    assert not torch.isnan(out).any()
    off = 0
    for idx, n in enumerate(lens[:4]):  # spot-check the skewed head + a few tails
        qi = q[off : off + n].cpu().unsqueeze(0)
        ki = k[off : off + n].cpu().unsqueeze(0)
        vi = v[off : off + n].cpu().unsqueeze(0)
        out_ref, out_pt = _ref_pair(qi, ki, vi, causal=True)
        check_tensor_budget(
            "out",
            out[off : off + n].unsqueeze(0),
            out_ref,
            out_pt,
            dtype,
            detail=f"skew seq{idx} n={n}",
        )
        off += n


# ---------------------------------------------------------------------------
# heuristic behavior (unit)
# ---------------------------------------------------------------------------


def test_heuristic_extreme_skew_picks_loop():
    lens = [8192] + [64] * 63
    assert not _prefer_batched(64, lens, lens, loop_hits_sdpa=False, grad_mode=False)
    assert not _prefer_batched(64, lens, lens, loop_hits_sdpa=True, grad_mode=True)


def test_heuristic_launch_bound_picks_batched():
    lens = [128] * 64
    assert _prefer_batched(64, lens, lens, loop_hits_sdpa=False, grad_mode=False)
    lens = [32] * 256
    assert _prefer_batched(256, lens, lens, loop_hits_sdpa=True, grad_mode=False)


def test_heuristic_moderate_ragged_lse_picks_batched():
    torch.manual_seed(0)
    lens = torch.randint(64, 1025, (32,)).tolist()
    assert _prefer_batched(32, lens, lens, loop_hits_sdpa=False, grad_mode=False)


def test_heuristic_sdpa_inference_large_seqs_picks_loop():
    # per-sequence is_causal SDPA launches beat masked batched SDPA here
    lens = [2048] * 8
    assert not _prefer_batched(8, lens, lens, loop_hits_sdpa=True, grad_mode=False)
