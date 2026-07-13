"""Seam A tests: the FA2 public API (``flash_attn.flash_attn_interface``) on MPS.

Exercises the five-function adapter (``flash_attn/mps/fa2_backend.py``) through
the real public entry points — ``flash_attn_func``, ``flash_attn_varlen_func``,
the packed variants, ``flash_attn_with_kvcache`` — exactly the way third-party
code calls them. CPU is the oracle (``flash_attn/cute/testing.py::attention_ref``,
fp32-upcast), MPS is the defendant, budgets are the cute suite's own.

The specifically FA2-shaped contracts pinned here:

- ``dq``/``dk``/``dv`` are filled IN PLACE through preallocated tensors,
  including the non-contiguous ``dqkv[:, :, i]`` views of the packed variants
  (getting this wrong yields silently-zero gradients).
- ``lse`` for fully-masked rows is ``+inf`` (FA2 CUDA convention), not the
  core's native ``-inf``.
- ``window_size`` uses the FA2 encoding: negative == infinite.
- perf knobs (``deterministic``, ``num_splits``) are accepted and ignored;
  unsupported features (dropout, paged KV, rotary-in-kvcache,
  ``return_attn_probs`` S_dmask) raise ``NotImplementedError``.
"""

import math

import pytest
import torch

from checks import check_tensor_budget
from flash_attn import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)
from flash_attn.cute.testing import attention_ref

if not torch.backends.mps.is_available():
    pytest.skip("MPS not available", allow_module_level=True)

DEVICE = "mps"
DTYPES = [torch.float16, torch.bfloat16]


def _fa2_window_to_ref(window_size):
    """FA2 (-1 == infinite) -> attention_ref (None == infinite)."""
    return tuple(None if w is not None and w < 0 else w for w in window_size)


def _make_qkv(batch, sq, sk, h, h_kv, d, dtype, seed):
    torch.manual_seed(seed)
    q = torch.randn(batch, sq, h, d).to(dtype)
    k = torch.randn(batch, sk, h_kv, d).to(dtype)
    v = torch.randn(batch, sk, h_kv, d).to(dtype)
    return q, k, v


def _oracle(q, k, v, g, *, causal, window_size, softcap=0.0, attn_bias=None):
    """(out_ref, grads_ref, out_pt, grads_pt) on CPU; ref fp32-upcast, pt in-dtype."""
    ref = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    pt = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    kwargs = dict(
        causal=causal,
        window_size=_fa2_window_to_ref(window_size),
        softcap=softcap,
        attn_bias=attn_bias,
    )
    out_ref, _ = attention_ref(*ref, None, None, **kwargs)
    out_pt, _ = attention_ref(*pt, None, None, upcast=False, reorder_ops=True, **kwargs)
    dref = torch.autograd.grad(out_ref, ref, g.to(out_ref.dtype))
    dpt = torch.autograd.grad(out_pt, pt, g)
    return out_ref, dref, out_pt, dpt


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mha", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize(
    "window_size,softcap",
    [((-1, -1), 0.0), ((17, -1), 0.0), ((-1, -1), 30.0)],
    ids=["plain", "local", "softcap"],
)
def test_fa2_flash_attn_func(dtype, causal, mha, window_size, softcap):
    batch, sq, sk, h, d = 2, 113, 203, 8, 64
    h_kv = {"mha": 8, "gqa": 2, "mqa": 1}[mha]
    q, k, v = _make_qkv(batch, sq, sk, h, h_kv, d, dtype, seed=11)
    g = torch.randn(batch, sq, h, d).to(dtype)
    out_ref, dref, out_pt, dpt = _oracle(
        q, k, v, g, causal=causal, window_size=window_size, softcap=softcap
    )

    qm = q.detach().to(DEVICE).requires_grad_()
    km = k.detach().to(DEVICE).requires_grad_()
    vm = v.detach().to(DEVICE).requires_grad_()
    out = flash_attn_func(
        qm, km, vm, causal=causal, window_size=window_size, softcap=softcap
    )
    assert out.dtype == dtype and out.shape == (batch, sq, h, d)
    dq, dk, dv = torch.autograd.grad(out, (qm, km, vm), g.to(DEVICE))

    rtol = 2.0 if softcap == 0.0 else 3.0
    extra = 0.0 if softcap == 0.0 else 3e-4
    detail = f"fa2 {mha} causal={causal} win={window_size} cap={softcap}"
    check_tensor_budget("out", out, out_ref, out_pt, dtype, rtol=rtol, detail=detail)
    for name, a, r, p in [
        ("dq", dq, dref[0], dpt[0]),
        ("dk", dk, dref[1], dpt[1]),
        ("dv", dv, dref[2], dpt[2]),
    ]:
        check_tensor_budget(
            name, a, r, p, dtype, rtol=rtol, extra_atol=extra, detail=detail
        )


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
def test_fa2_alibi(dtype):
    batch, sq, sk, h, d = 2, 128, 151, 8, 64
    torch.manual_seed(12)
    slopes = torch.rand(batch, h, dtype=torch.float32) * 0.3
    q, k, v = _make_qkv(batch, sq, sk, h, h, d, dtype, seed=12)
    g = torch.randn(batch, sq, h, d).to(dtype)
    row = torch.arange(sq, dtype=torch.long).unsqueeze(-1)
    col = torch.arange(sk, dtype=torch.long)
    bias = -slopes.view(batch, h, 1, 1) * (row + sk - sq - col).abs().float()
    out_ref, dref, out_pt, dpt = _oracle(
        q, k, v, g, causal=False, window_size=(-1, -1), attn_bias=bias
    )
    qm = q.detach().to(DEVICE).requires_grad_()
    km = k.detach().to(DEVICE).requires_grad_()
    vm = v.detach().to(DEVICE).requires_grad_()
    out = flash_attn_func(qm, km, vm, alibi_slopes=slopes.to(DEVICE))
    dq, dk, dv = torch.autograd.grad(out, (qm, km, vm), g.to(DEVICE))
    detail = "fa2 alibi"
    check_tensor_budget("out", out, out_ref, out_pt, dtype, detail=detail)
    check_tensor_budget("dq", dq, dref[0], dpt[0], dtype, detail=detail)
    check_tensor_budget("dk", dk, dref[1], dpt[1], dtype, detail=detail)
    check_tensor_budget("dv", dv, dref[2], dpt[2], dtype, detail=detail)


@pytest.mark.parametrize("packed", ["qkv", "kv"])
def test_fa2_packed_inplace_grads(packed):
    """The packed variants pass dkv[:, :, i] views into bwd — the gradients must
    land through those non-contiguous views (a wrong bwd yields all-zero dkv)."""
    dtype = torch.float16
    batch, seqlen, h, d = 2, 96, 4, 64
    torch.manual_seed(13)
    if packed == "qkv":
        qkv = (
            torch.randn(batch, seqlen, 3, h, d, device=DEVICE)
            .to(dtype)
            .requires_grad_()
        )
        out = flash_attn_qkvpacked_func(qkv, causal=True)
        g = torch.randn_like(out)
        out.backward(g)
        grad = qkv.grad
        ref_in = qkv.detach().cpu().float().requires_grad_()
        out_ref, _ = attention_ref(
            ref_in[:, :, 0],
            ref_in[:, :, 1],
            ref_in[:, :, 2],
            None,
            None,
            causal=True,
            upcast=False,
        )
        out_ref.backward(g.cpu().float())
        grad_ref = ref_in.grad
    else:
        q = torch.randn(batch, seqlen, h, d, device=DEVICE).to(dtype).requires_grad_()
        kv = (
            torch.randn(batch, seqlen, 2, h, d, device=DEVICE)
            .to(dtype)
            .requires_grad_()
        )
        out = flash_attn_kvpacked_func(q, kv, causal=True)
        g = torch.randn_like(out)
        out.backward(g)
        grad = kv.grad
        qr = q.detach().cpu().float().requires_grad_()
        kvr = kv.detach().cpu().float().requires_grad_()
        out_ref, _ = attention_ref(
            qr, kvr[:, :, 0], kvr[:, :, 1], None, None, causal=True, upcast=False
        )
        out_ref.backward(g.cpu().float())
        grad_ref = kvr.grad
    assert grad is not None and (grad != 0).any(), (
        "packed grads must not be silently zero"
    )
    assert torch.isfinite(grad).all()
    diff = (grad.cpu().float() - grad_ref).abs().max().item()
    assert diff < 5e-3, f"packed grad mismatch vs fp32 oracle: {diff}"


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True])
def test_fa2_varlen(dtype, causal):
    torch.manual_seed(14)
    lens_q = [37, 128, 1, 63]
    lens_k = [64, 100, 17, 63]
    h, d = 4, 64
    cu_q = torch.tensor(
        [0] + torch.tensor(lens_q).cumsum(0).tolist(), dtype=torch.int32
    )
    cu_k = torch.tensor(
        [0] + torch.tensor(lens_k).cumsum(0).tolist(), dtype=torch.int32
    )
    tq, tk = sum(lens_q), sum(lens_k)
    q = torch.randn(tq, h, d).to(dtype)
    k = torch.randn(tk, h, d).to(dtype)
    v = torch.randn(tk, h, d).to(dtype)
    g = torch.randn(tq, h, d).to(dtype)

    qm = q.detach().to(DEVICE).requires_grad_()
    km = k.detach().to(DEVICE).requires_grad_()
    vm = v.detach().to(DEVICE).requires_grad_()
    out = flash_attn_varlen_func(
        qm,
        km,
        vm,
        cu_q.to(DEVICE),
        cu_k.to(DEVICE),
        max(lens_q),
        max(lens_k),
        causal=causal,
    )
    dq, dk, dv = torch.autograd.grad(out, (qm, km, vm), g.to(DEVICE))
    assert out.shape == (tq, h, d)

    for i in range(len(lens_q)):
        qs, qe = int(cu_q[i]), int(cu_q[i + 1])
        ks, ke = int(cu_k[i]), int(cu_k[i + 1])
        seq = [
            t[s:e].unsqueeze(0).detach().clone().requires_grad_()
            for t, (s, e) in ((q, (qs, qe)), (k, (ks, ke)), (v, (ks, ke)))
        ]
        pt = [t.detach().clone().requires_grad_() for t in seq]
        out_ref, _ = attention_ref(*seq, None, None, causal=causal)
        out_pt, _ = attention_ref(
            *pt, None, None, causal=causal, upcast=False, reorder_ops=True
        )
        gi = g[qs:qe].unsqueeze(0)
        dref = torch.autograd.grad(out_ref, seq, gi.to(out_ref.dtype))
        dpt = torch.autograd.grad(out_pt, pt, gi)
        detail = f"fa2 varlen seq{i} causal={causal}"
        check_tensor_budget(
            "out", out[qs:qe], out_ref[0], out_pt[0], dtype, detail=detail
        )
        check_tensor_budget(
            "dq", dq[qs:qe], dref[0][0], dpt[0][0], dtype, detail=detail
        )
        check_tensor_budget(
            "dk", dk[ks:ke], dref[1][0], dpt[1][0], dtype, detail=detail
        )
        check_tensor_budget(
            "dv", dv[ks:ke], dref[2][0], dpt[2][0], dtype, detail=detail
        )


def test_fa2_lse_masked_rows_plus_inf():
    """causal with seqlen_q > seqlen_k leaves the first rows fully masked; the
    FA2 contract is lse == +inf there (CUDA csrc/flash_attn/src/softmax.h:180),
    the opposite sign of the core's/FA4's -inf."""
    torch.manual_seed(15)
    q = torch.randn(1, 64, 4, 64, device=DEVICE, dtype=torch.float16)
    k = torch.randn(1, 32, 4, 64, device=DEVICE, dtype=torch.float16)
    v = torch.randn(1, 32, 4, 64, device=DEVICE, dtype=torch.float16)
    out, lse, _ = flash_attn_func(q, k, v, causal=True, return_attn_probs=True)
    n_masked = 64 - 32
    assert (lse[:, :, :n_masked] == float("inf")).all(), (
        "FA2 masked-row lse must be +inf"
    )
    assert torch.isfinite(lse[:, :, n_masked:]).all()
    assert (out[:, :n_masked] == 0).all()
    assert not torch.isnan(out).any()


def test_fa2_kvcache():
    torch.manual_seed(16)
    dtype = torch.float16
    batch, s_cache, s_new, h, d = 2, 64, 4, 4, 64
    k_cache = torch.randn(batch, s_cache, h, d, device=DEVICE).to(dtype)
    v_cache = torch.randn(batch, s_cache, h, d, device=DEVICE).to(dtype)
    k_cache_orig, v_cache_orig = k_cache.clone(), v_cache.clone()
    q = torch.randn(batch, s_new, h, d, device=DEVICE).to(dtype)
    k_new = torch.randn(batch, s_new, h, d, device=DEVICE).to(dtype)
    v_new = torch.randn(batch, s_new, h, d, device=DEVICE).to(dtype)
    seqlens = torch.tensor([30, 50], dtype=torch.int32, device=DEVICE)
    out, lse = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k_new,
        v_new,
        cache_seqlens=seqlens,
        causal=True,
        return_softmax_lse=True,
    )
    assert lse.shape == (batch, h, s_new) and lse.dtype == torch.float32
    for i in range(batch):
        L = int(seqlens[i])
        assert (k_cache[i, L : L + s_new] == k_new[i]).all(), (
            "cache must be updated in place"
        )
        kf = torch.cat([k_cache_orig[i, :L], k_new[i]], 0).unsqueeze(0).cpu()
        vf = torch.cat([v_cache_orig[i, :L], v_new[i]], 0).unsqueeze(0).cpu()
        qi = q[i : i + 1].cpu()
        out_ref, _ = attention_ref(qi, kf, vf, None, None, causal=True)
        out_pt, _ = attention_ref(
            qi, kf, vf, None, None, causal=True, upcast=False, reorder_ops=True
        )
        check_tensor_budget(
            "out", out[i : i + 1], out_ref, out_pt, dtype, detail=f"kvcache b{i}"
        )


def test_fa2_kvcache_leftpad_and_batch_idx():
    torch.manual_seed(17)
    dtype = torch.float16
    batch, s_cache, h, d = 2, 48, 4, 64
    k_cache = torch.randn(batch, s_cache, h, d, device=DEVICE).to(dtype)
    v_cache = torch.randn(batch, s_cache, h, d, device=DEVICE).to(dtype)
    q = torch.randn(batch, 1, h, d, device=DEVICE).to(dtype)
    seqlens = torch.tensor([40, 33], dtype=torch.int32, device=DEVICE)
    leftpad = torch.tensor([5, 0], dtype=torch.int32, device=DEVICE)
    batch_idx = torch.tensor([1, 0], dtype=torch.int32, device=DEVICE)
    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=seqlens,
        cache_leftpad=leftpad,
        cache_batch_idx=batch_idx,
        causal=True,
    )
    for i in range(batch):
        bi, L, lp = int(batch_idx[i]), int(seqlens[i]), int(leftpad[i])
        kf = k_cache[bi, lp:L].unsqueeze(0).cpu()
        vf = v_cache[bi, lp:L].unsqueeze(0).cpu()
        out_ref, _ = attention_ref(q[i : i + 1].cpu(), kf, vf, None, None, causal=True)
        out_pt, _ = attention_ref(
            q[i : i + 1].cpu(),
            kf,
            vf,
            None,
            None,
            causal=True,
            upcast=False,
            reorder_ops=True,
        )
        check_tensor_budget(
            "out",
            out[i : i + 1],
            out_ref,
            out_pt,
            dtype,
            detail=f"kvcache leftpad b{i}",
        )


def test_fa2_perf_knobs_accepted():
    """deterministic / num_splits are perf knobs, not semantics: must not raise."""
    q = torch.randn(
        1, 32, 2, 64, device=DEVICE, dtype=torch.float16, requires_grad=True
    )
    k = torch.randn(1, 32, 2, 64, device=DEVICE, dtype=torch.float16)
    v = torch.randn(1, 32, 2, 64, device=DEVICE, dtype=torch.float16)
    out = flash_attn_func(q, k, v, causal=True, deterministic=True)
    out.sum().backward()
    assert q.grad is not None
    flash_attn_with_kvcache(q.detach(), k, v, num_splits=4)


def test_fa2_unsupported_raise():
    q = torch.randn(1, 8, 2, 64, device=DEVICE, dtype=torch.float16)
    k = torch.randn(1, 8, 2, 64, device=DEVICE, dtype=torch.float16)
    v = torch.randn(1, 8, 2, 64, device=DEVICE, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="dropout"):
        flash_attn_func(q, k, v, dropout_p=0.17)
    cu = torch.tensor([0, 8], dtype=torch.int32, device=DEVICE)
    with pytest.raises(NotImplementedError, match="block_table|paged"):
        flash_attn_varlen_func(
            q[0],
            k[0],
            v[0],
            cu,
            cu,
            8,
            8,
            block_table=torch.zeros(1, 1, dtype=torch.int32, device=DEVICE),
        )
    rotary = torch.randn(16, 16, device=DEVICE, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="rotary"):
        flash_attn_with_kvcache(
            q,
            k,
            v,
            k=torch.randn_like(q),
            v=torch.randn_like(q),
            cache_seqlens=torch.zeros(1, dtype=torch.int32, device=DEVICE),
            rotary_cos=rotary,
            rotary_sin=rotary,
        )


def test_fa2_softmax_scale():
    torch.manual_seed(18)
    dtype = torch.float16
    d = 64
    scale = 0.25  # scale * sqrt(d) == 2.0, lossless pre-scaling for the oracle
    q, k, v = _make_qkv(2, 64, 80, 4, 4, d, dtype, seed=18)
    q_scaled = (q.float() * (scale * math.sqrt(d))).to(dtype)
    out_ref, _ = attention_ref(q_scaled, k, v, None, None, causal=True)
    out_pt, _ = attention_ref(
        q_scaled, k, v, None, None, causal=True, upcast=False, reorder_ops=True
    )
    out = flash_attn_func(
        q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), softmax_scale=scale, causal=True
    )
    check_tensor_budget("out", out, out_ref, out_pt, dtype, detail="fa2 custom scale")
