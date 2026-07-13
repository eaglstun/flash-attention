"""Phase 3b split-path regressions: the SDPA-out + streaming-lse forward and
the SDPA-powered chunked recompute backward.

The parity suite (test_attention_parity.py) already gates the split paths
against the CPU/fp64 oracle — fp16/bf16 route through them by default. This
file pins what the parity grid does not: split-vs-chunked agreement on the
same device, the fp32 dtype gate, and the SDPA-autograd INT_MAX guard.
"""

import pytest
import torch

import flash_attn.mps.core as core
from flash_attn.mps.core import mps_flash_attn_func

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def _run(q, k, v, g, **kw):
    ql = q.detach().requires_grad_()
    kl = k.detach().requires_grad_()
    vl = v.detach().requires_grad_()
    out, lse = mps_flash_attn_func(ql, kl, vl, **kw)
    dq, dk, dv = torch.autograd.grad(out, (ql, kl, vl), g)
    return out, lse, dq, dk, dv


CASES = {
    "causal": dict(sq=257, sk=257, causal=True),
    "causal_uneq": dict(sq=113, sk=203, causal=True),
    "causal_q_gt_k": dict(sq=128, sk=64, causal=True),  # fully-masked rows
    "window": dict(sq=247, sk=311, window_size=(37, 53)),
    "causal_window_left": dict(sq=247, sk=311, causal=True, window_size=(64, None)),
    "seqused": dict(sq=257, sk=311, causal=True, seqused_k=[100, 311]),
    "leftpad": dict(
        sq=64, sk=311, causal=True, seqused_k=[200, 311], key_leftpad=[5, 0]
    ),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("case", CASES.keys())
def test_split_matches_chunked(device, dtype, case):
    """The split forward (SDPA out + streaming lse) and the SDPA recompute
    backward must agree with the chunked core / manual recompute within the
    dtype's storage rounding. lse is fp32 both ways and must agree tightly;
    inf positions (fully-masked rows) must match exactly."""
    spec = dict(CASES[case])
    sq, sk = spec.pop("sq"), spec.pop("sk")
    for name in ("seqused_k", "key_leftpad"):
        if name in spec:
            spec[name] = torch.tensor(spec[name], device=device)
    torch.manual_seed(0)
    batch, h, hkv, d = 2, 8, 2, 64
    q = torch.randn(batch, sq, h, d, device=device).to(dtype)
    k = torch.randn(batch, sk, hkv, d, device=device).to(dtype)
    v = torch.randn(batch, sk, hkv, d, device=device).to(dtype)
    g = torch.randn(batch, sq, h, d, device=device).to(dtype)

    split = _run(q, k, v, g, **spec)
    chunked = _run(q, k, v, g, _force_chunked=True, **spec)

    tol = 16 * torch.finfo(dtype).eps  # storage rounding, both sides fp32 math
    for name, a, b in zip(["out", "lse", "dq", "dk", "dv"], split, chunked):
        inf_a, inf_b = torch.isinf(a), torch.isinf(b)
        assert torch.equal(inf_a, inf_b), f"{name}: inf positions differ"
        fin = ~inf_a
        assert not torch.isnan(a).any(), f"{name}: NaN in split path"
        t = 1e-5 if name == "lse" else tol  # lse is fp32 on both paths
        if fin.any():
            diff = (a[fin].float() - b[fin].float()).abs().max().item()
            assert diff <= t, f"{name}: split vs chunked diff {diff:.3e} > {t:g}"


@pytest.mark.parametrize("device", DEVICES)
def test_fp32_stays_off_the_kernels(device, monkeypatch):
    """The fp32 dtype gate: fp32 lse-producing calls must never route into
    _sdpa_out/_lse_forward (their kernels are not cross-device deterministic
    at fp32's ulp-scale budgets); fp16 must."""
    calls = {"n": 0}
    real = core._lse_forward

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(core, "_lse_forward", counting)
    q32 = torch.randn(1, 64, 2, 32, device=device)
    out, lse = mps_flash_attn_func(q32, q32, q32, causal=True)
    assert calls["n"] == 0, "fp32 must stay on the chunked core"
    q16 = q32.half()
    out, lse = mps_flash_attn_func(q16, q16, q16, causal=True)
    assert calls["n"] == 1, "fp16 must take the split forward"


@pytest.mark.parametrize("device", DEVICES)
def test_sdpa_autograd_guard_routes_to_chunked_backward(device, monkeypatch):
    """Past _SDPA_AUTOGRAD_MAX_ELEMENTS the no-lse grad-mode fast path must
    route into MPSFlashAttnFunc (memory-flat recompute backward) instead of
    letting SDPA's autograd materialize the O(s^2) score matrix — that is
    the 16k INT_MAX hard-fail on MPS. Pinned with a shrunken cap so the test
    stays small; gradients must match the direct path."""
    torch.manual_seed(3)
    b, s, h, d = 2, 192, 4, 32
    q = torch.randn(b, s, h, d, device=device).half()
    k = torch.randn(b, s, h, d, device=device).half()
    v = torch.randn(b, s, h, d, device=device).half()
    g = torch.randn(b, s, h, d, device=device).half()

    direct = _run(q, k, v, g, causal=True, return_lse=False)
    monkeypatch.setattr(core, "_SDPA_AUTOGRAD_MAX_ELEMENTS", 1)
    guarded = _run(q, k, v, g, causal=True, return_lse=False)
    assert guarded[1] is None  # still no fabricated lse
    for name, a, b_ in zip(
        ["out", "dq", "dk", "dv"], (guarded[0], *guarded[2:]), (direct[0], *direct[2:])
    ):
        diff = (a.float() - b_.float()).abs().max().item()
        assert diff <= 16 * torch.finfo(torch.float16).eps, f"{name}: {diff:.3e}"


@pytest.mark.parametrize("device", DEVICES)
def test_tensor_window_sizes_match_ints(device):
    """Window sizes arriving as 0-dim tensors (the FA2 suite passes
    torch.randint results; the CUDA extension's pybind coerces to int) must
    behave exactly like ints. Pinned because x[lo:hi] with a tensor bound
    silently returns the UNSLICED tensor, which made the recompute
    backward's K-truncation evaporate while its masks assumed the truncated
    width — wrong gradients, caught by test_flash_attn_qkvpacked on MPS."""
    torch.manual_seed(6)
    b, s, h, d = 2, 640, 4, 64
    q = torch.randn(b, s, h, d, device=device).half()
    k = torch.randn(b, s, h, d, device=device).half()
    v = torch.randn(b, s, h, d, device=device).half()
    g = torch.randn(b, s, h, d, device=device).half()
    w = (137, 211)
    w_t = (torch.tensor(w[0]), torch.tensor(w[1]))
    ints = _run(q, k, v, g, window_size=w)
    tens = _run(q, k, v, g, window_size=w_t)
    for name, a, b_ in zip(["out", "lse", "dq", "dk", "dv"], ints, tens):
        inf_a = torch.isinf(a)
        assert torch.equal(inf_a, torch.isinf(b_)), f"{name}: inf positions differ"
        fin = ~inf_a
        assert torch.equal(a[fin], b_[fin]), (
            f"{name}: tensor-window result differs from int"
        )


def test_single_key_grads_exact_zero_through_default_path():
    """seqlen_k == 1 keeps the manual recompute: single-key softmax is the
    constant 1, so dq and dk are exactly 0 (SDPA's backward would leave
    ~1-ulp noise there). Runs on the default device routing."""
    device = DEVICES[-1]
    torch.manual_seed(4)
    q = torch.randn(1, 339, 6, 64, device=device).half()
    k = torch.randn(1, 1, 6, 64, device=device).half()
    v = torch.randn(1, 1, 6, 64, device=device).half()
    g = torch.randn(1, 339, 6, 64, device=device).half()
    _, _, dq, dk, dv = _run(q, k, v, g)
    assert (dq == 0).all() and (dk == 0).all()
    assert torch.isfinite(dv).all()
