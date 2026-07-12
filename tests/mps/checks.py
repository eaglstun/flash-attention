"""Error-budget assertions for the MPS parity suite.

Reuses the exact assertion style of tests/cute/test_flash_attn.py (line ~375
and ``check_tensor_vs_ref``): the implementation under test may be at most
``rtol`` times as wrong as a plain PyTorch implementation run in the test
dtype, plus a noise-floor atol derived from the reference itself:

    (out - out_ref).abs().max() <= rtol * (out_pt - out_ref).abs().max() + atol
    atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max()   (+ extra_atol)

with rtol = 2 (3 when softcap != 0, and grads get an extra 3e-4 atol then).

CPU is the oracle, MPS is the defendant: ``ref`` is the fp32-upcast oracle,
``pt`` the same math in the test dtype, both on CPU; ``actual`` comes from the
MPS core. Infinities (lse of fully-masked rows) must match exactly and are
excluded from the finite comparison.
"""

import torch

# {f"{dtype}/{tensor}": (worst_ratio, "test detail")}
WORST_RATIOS = {}


def _record(dtype, name, ratio, detail):
    key = f"{dtype}/{name}"
    if key not in WORST_RATIOS or ratio > WORST_RATIOS[key][0]:
        WORST_RATIOS[key] = (ratio, detail)


def check_tensor_budget(
    name, actual, ref, pt, dtype, rtol=2.0, extra_atol=0.0, detail=""
):
    """Assert `actual` is within the FA error budget of `ref`, using `pt` as the baseline."""
    assert actual is not None
    actual = actual.detach().float().cpu()
    ref = ref.detach().float().cpu()
    pt = pt.detach().float().cpu()
    assert actual.shape == ref.shape, (
        f"{name}: shape {tuple(actual.shape)} != ref {tuple(ref.shape)}"
    )
    assert not torch.isnan(actual).any(), f"{name}: NaN in output"
    assert not torch.isnan(ref).any(), f"{name}: NaN in reference (bad test case)"

    inf_ref = torch.isinf(ref)
    inf_act = torch.isinf(actual)
    assert torch.equal(inf_act, inf_ref), f"{name}: inf positions differ from reference"
    if inf_ref.any():
        assert torch.equal(actual[inf_act], ref[inf_ref]), (
            f"{name}: inf signs differ from reference"
        )
    finite = ~inf_ref
    if finite.sum() == 0:
        return
    actual, ref, pt = actual[finite], ref[finite], pt[finite]

    # Noise floor of the reference, at the resolution of the test dtype (the
    # same trick as the cute suite, whose ref tensors are stored in the test
    # dtype: fp16/bf16 get their own rounding noise as the floor). This is
    # what absorbs e.g. an fp16-stored reference lse compared against our
    # exact fp32 lse.
    ref_td = ref.to(dtype)
    atol = 2 * (ref_td + 0.3 - 0.3 - ref_td).abs().max().float().item() + extra_atol
    diff_max = (actual - ref).abs().max().item()
    diff_pt_max = (pt - ref).abs().max().item()
    budget = rtol * diff_pt_max + atol
    ratio = (
        diff_max / budget if budget > 0 else (0.0 if diff_max == 0 else float("inf"))
    )
    _record(
        str(dtype).replace("torch.", ""),
        name,
        ratio,
        f"{detail} diff={diff_max:.3e} pt_diff={diff_pt_max:.3e} atol={atol:.3e}",
    )
    assert diff_max <= budget, (
        f"{name}: max diff {diff_max:.3e} exceeds budget {budget:.3e} "
        f"(rtol={rtol} * pt_diff={diff_pt_max:.3e} + atol={atol:.3e}) [{detail}]"
    )
