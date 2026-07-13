"""Isolate the torch-MPS <-> MLX bridge cost from the kernel cost.

Findings this script encodes (verified empirically, torch 2.13 / mlx 0.32):

- torch MPS tensors export DLPack with device ``kDLMetal``.
- ``mx.from_dlpack(torch_tensor)`` is ZERO-COPY (shared unified memory);
  ``mx.array(torch_tensor)`` COPIES.
- ``torch.from_dlpack(mlx_array)`` is zero-copy back onto ``mps:0``.

The remaining real costs of a per-call bridge are therefore: (a) the
transpose+``.contiguous()`` torch launch to get (b,h,s,d) row-major, (b) the
``torch.mps.synchronize()`` needed before MLX may touch the shared buffer,
(c) MLX graph build + ``mx.eval`` scheduling overhead. This script measures
each leg on realistic q/k/v sizes.

Usage: python benchmarks/mps/bench_interop.py
"""

import sys

import torch

sys.path.insert(0, "benchmarks/mps")
from common import median_iqr, sync, time_op  # noqa: E402

import mlx.core as mx  # noqa: E402


def bench(label, fn, **kw):
    times = time_op(fn, **kw)
    med, iqr = median_iqr(times)
    print(f"  {label:52s} {med * 1e6:10.1f} us  (iqr {iqr * 1e6:.1f}, n={len(times)})")
    return med


def main():
    print("torch", torch.__version__, "| mlx", mx.__version__)
    for b, s, h, d in [(8, 2048, 8, 64), (1, 16384, 8, 128)]:
        nbytes = b * s * h * d * 2 * 3  # q,k,v fp16
        print(
            f"\nshape b={b} s={s} h={h} d={d} (q+k+v = {nbytes / 2**20:.0f} MiB fp16)"
        )
        q = torch.randn(b, s, h, d, device="mps", dtype=torch.float16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        scale = d**-0.5

        # leg (a): torch-side layout fix (b,s,h,d) -> contiguous (b,h,s,d)
        def leg_a():
            for t in (q, k, v):
                t.transpose(1, 2).contiguous()

        bench("(a) transpose+contiguous q,k,v (torch)", leg_a)

        # leg (b)+(c) import: sync + zero-copy dlpack import
        qc = q.transpose(1, 2).contiguous()
        kc = k.transpose(1, 2).contiguous()
        vc = v.transpose(1, 2).contiguous()
        sync()

        def leg_import():
            arrs = [mx.from_dlpack(t) for t in (qc, kc, vc)]
            mx.eval(*arrs)

        bench("(b) mx.from_dlpack q,k,v (zero-copy) + eval", leg_import)

        def leg_import_copy():
            arrs = [mx.array(t) for t in (qc, kc, vc)]
            mx.eval(*arrs)

        bench("(b') mx.array q,k,v (copying) + eval", leg_import_copy)

        # kernel only, arrays resident
        qm, km, vm = (mx.from_dlpack(t) for t in (qc, kc, vc))
        mx.eval(qm, km, vm)

        def kernel_only():
            o = mx.fast.scaled_dot_product_attention(
                qm, km, vm, scale=scale, mask="causal"
            )
            mx.eval(o)

        bench("(k) mx.fast.sdpa kernel only (causal)", kernel_only)

        # export leg
        om = mx.fast.scaled_dot_product_attention(
            qm, km, vm, scale=scale, mask="causal"
        )
        mx.eval(om)

        def leg_export():
            torch.from_dlpack(om).transpose(1, 2)

        bench("(c) torch.from_dlpack(out) (zero-copy)", leg_export)

        # the whole bridged call, as bench_attention times it
        def bridged():
            qc = q.transpose(1, 2).contiguous()
            kc = k.transpose(1, 2).contiguous()
            vc = v.transpose(1, 2).contiguous()
            sync()
            o = mx.fast.scaled_dot_product_attention(
                mx.from_dlpack(qc),
                mx.from_dlpack(kc),
                mx.from_dlpack(vc),
                scale=scale,
                mask="causal",
            )
            mx.eval(o)
            torch.from_dlpack(o).transpose(1, 2)

        bench("(total) bridged sdpa call", bridged)


if __name__ == "__main__":
    main()
