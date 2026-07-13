"""The known-pathological Phase 1 paths: varlen's per-sequence Python loop and
the per-batch kvcache decode loop, quantified against batched alternatives.

- varlen: ``flash_attn_varlen_func`` (FA2 seam) runs one core call per packed
  sequence. Alternative measured: pad to a dense batch and run one batched
  call (core, and torch SDPA) -- extra FLOPs on padding, but one launch.
- decode: ``flash_attn_with_kvcache`` loops per batch element. Alternative:
  one batched SDPA over the whole cache.

Usage: PYTHONPATH=. python benchmarks/mps/bench_varlen.py
"""

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "benchmarks/mps")
from common import median_iqr, sync, time_op  # noqa: E402

from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache  # noqa: E402


def bench(label, fn, **kw):
    times = time_op(fn, **kw)
    med, iqr = median_iqr(times)
    print(f"  {label:56s} {med * 1e3:9.2f} ms (iqr {iqr * 1e3:.2f}, n={len(times)})")
    return med


def varlen_bench():
    torch.manual_seed(0)
    nseq, hq, hk, hd = 32, 8, 8, 64
    lens = torch.randint(64, 1025, (nseq,))
    total = int(lens.sum())
    max_len = int(lens.max())
    cu = torch.zeros(nseq + 1, dtype=torch.int32)
    cu[1:] = lens.cumsum(0)
    cu_mps = cu.to("mps")
    print(
        f"\n== varlen: {nseq} packed seqs, lens 64..1024 (total={total}, max={max_len}), "
        f"h{hq} d{hd} fp16 causal =="
    )
    q = torch.randn(
        total, hq, hd, device="mps", dtype=torch.float16, requires_grad=True
    )
    k = torch.randn(
        total, hk, hd, device="mps", dtype=torch.float16, requires_grad=True
    )
    v = torch.randn(
        total, hk, hd, device="mps", dtype=torch.float16, requires_grad=True
    )
    dout = torch.randn_like(q)

    def varlen_fwd():
        with torch.no_grad():
            flash_attn_varlen_func(
                q, k, v, cu_mps, cu_mps, max_len, max_len, causal=True
            )

    bench("flash_attn_varlen_func fwd (per-seq loop)", varlen_fwd, budget_s=20)

    def varlen_fwd_bwd():
        q.grad = k.grad = v.grad = None
        out = flash_attn_varlen_func(
            q, k, v, cu_mps, cu_mps, max_len, max_len, causal=True
        )
        out.backward(dout)

    bench(
        "flash_attn_varlen_func fwd+bwd (per-seq loop)",
        varlen_fwd_bwd,
        warmup=2,
        iters=5,
        budget_s=60,
    )

    # padded-dense alternatives (upper bound on wasted FLOPs, one launch)
    qp = torch.zeros(nseq, max_len, hq, hd, device="mps", dtype=torch.float16)
    kp = torch.zeros(nseq, max_len, hk, hd, device="mps", dtype=torch.float16)
    vp = torch.zeros(nseq, max_len, hk, hd, device="mps", dtype=torch.float16)
    with torch.no_grad():
        for i in range(nseq):
            s0, s1 = int(cu[i]), int(cu[i + 1])
            qp[i, : s1 - s0] = q[s0:s1]
            kp[i, : s1 - s0] = k[s0:s1]
            vp[i, : s1 - s0] = v[s0:s1]
    qp.requires_grad_(), kp.requires_grad_(), vp.requires_grad_()
    doutp = torch.randn_like(qp)

    def padded_core_fwd():
        with torch.no_grad():
            flash_attn_func(qp, kp, vp, causal=True)

    bench(
        "padded dense core fwd (1 launch, wasted pad FLOPs)",
        padded_core_fwd,
        budget_s=20,
    )

    def padded_sdpa_fwd():
        with torch.no_grad():
            F.scaled_dot_product_attention(
                qp.transpose(1, 2),
                kp.transpose(1, 2),
                vp.transpose(1, 2),
                is_causal=True,
            )

    bench("padded dense torch-SDPA fwd", padded_sdpa_fwd, budget_s=20)

    def padded_sdpa_fwd_bwd():
        qp.grad = kp.grad = vp.grad = None
        o = F.scaled_dot_product_attention(
            qp.transpose(1, 2), kp.transpose(1, 2), vp.transpose(1, 2), is_causal=True
        ).transpose(1, 2)
        o.backward(doutp)

    bench("padded dense torch-SDPA fwd+bwd", padded_sdpa_fwd_bwd, budget_s=20)


def decode_bench():
    torch.manual_seed(0)
    b, cache_len, hq, hk, hd = 32, 4096, 8, 2, 128
    print(f"\n== kvcache decode: b={b} sq=1 cache={cache_len} h{hq}/{hk} d{hd} fp16 ==")
    q = torch.randn(b, 1, hq, hd, device="mps", dtype=torch.float16)
    kc = torch.randn(b, cache_len, hk, hd, device="mps", dtype=torch.float16)
    vc = torch.randn(b, cache_len, hk, hd, device="mps", dtype=torch.float16)
    cache_seqlens = torch.full((b,), cache_len, dtype=torch.int32, device="mps")

    def kvcache():
        flash_attn_with_kvcache(q, kc, vc, cache_seqlens=cache_seqlens, causal=True)

    bench("flash_attn_with_kvcache (per-batch loop)", kvcache, budget_s=20)

    def sdpa_decode():
        with torch.no_grad():
            F.scaled_dot_product_attention(
                q.transpose(1, 2),
                kc.transpose(1, 2),
                vc.transpose(1, 2),
                enable_gqa=True,
            )

    bench("batched torch-SDPA decode (1 launch)", sdpa_decode, budget_s=20)


if __name__ == "__main__":
    sync()
    varlen_bench()
    decode_bench()
