"""Shared helpers for the Apple Silicon (MPS) Phase 2 benchmarks.

Benchmark discipline (see docs/apple_silicon/BENCHMARKS.md):

- ``torch.mps.synchronize()`` before AND after every timed region. Without the
  trailing sync you time kernel *launch*, not execution. MLX regions
  ``mx.eval(...)`` their outputs, which blocks until the GPU work completes.
  (Do NOT add ``mx.synchronize()`` right after ``mx.eval`` on graphs that
  consume ``mx.from_dlpack``-imported torch-MPS buffers: mlx 0.32.0 can
  deadlock there, and ``mx.eval`` is already a completion barrier.)
- Warmup iterations first (MPS/MLX JIT-compile kernels on first use), then
  multiple timed iterations; report median and IQR, never a single run.
- Peak memory: torch.mps has no ``max_memory_allocated``; we use the
  ``driver_allocated_memory()`` high-water instead. The MPS caching allocator
  does not return memory to the driver until ``empty_cache()``, so
  (driver_allocated after run) - (driver_allocated after empty_cache, before
  run) is a faithful high-water proxy for the run. MLX reports
  ``mx.get_peak_memory()`` after ``mx.reset_peak_memory()``.
"""

import csv
import gc
import statistics
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


def sync() -> None:
    torch.mps.synchronize()


def time_op(
    fn: Callable[[], None],
    *,
    warmup: int = 3,
    iters: int = 10,
    budget_s: float = 30.0,
    min_iters: int = 3,
) -> List[float]:
    """Time ``fn`` with full sync discipline; returns per-iteration seconds.

    ``fn`` must itself leave any lazy work (MLX graphs) evaluated; torch-side
    completion is enforced here with ``torch.mps.synchronize()``.
    """
    for _ in range(warmup):
        fn()
    sync()
    times: List[float] = []
    t_start = time.perf_counter()
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
        if time.perf_counter() - t_start > budget_s and len(times) >= min_iters:
            break
    return times


def median_iqr(times: List[float]) -> tuple:
    med = statistics.median(times)
    if len(times) >= 4:
        qs = statistics.quantiles(times, n=4)
        iqr = qs[2] - qs[0]
    else:
        iqr = max(times) - min(times)
    return med, iqr


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------


class TorchMemHighWater:
    """Peak-allocated-memory proxy around a region.

    torch.mps has no ``max_memory_allocated``, and the driver high-water does
    not reliably reset with ``empty_cache()`` inside one process, so a
    background thread samples ``torch.mps.current_allocated_memory()`` (live
    tensor bytes, which DOES drop when intermediates are freed) at ~0.2 ms
    intervals for the duration of the region. Torch MPS ops allocate their
    output/intermediate buffers eagerly on the calling thread, so transient
    peaks are visible to the sampler even before the GPU executes. Peaks
    shorter than the sampling interval can be missed; treat the number as a
    lower bound with ~single-buffer resolution.
    """

    def __enter__(self):
        import threading

        gc.collect()
        torch.mps.empty_cache()
        sync()
        self.base = torch.mps.current_allocated_memory()
        self._peak = self.base
        self._stop = threading.Event()

        def sample():
            while not self._stop.is_set():
                cur = torch.mps.current_allocated_memory()
                if cur > self._peak:
                    self._peak = cur
                time.sleep(0.0002)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        sync()
        self._stop.set()
        self._thread.join(timeout=2)
        self._peak = max(self._peak, torch.mps.current_allocated_memory())
        self.peak_extra = self._peak - self.base
        return False


# ---------------------------------------------------------------------------
# FLOPs (flash-attention benchmark convention)
# ---------------------------------------------------------------------------


def attention_flops(
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads_q: int,
    head_dim: int,
    causal: bool,
    mode: str = "fwd",
) -> float:
    """4*b*h*sq*sk*d for forward (two matmuls, 2 FLOPs/MAC); x0.5 causal;
    bwd = 2.5x fwd; fwd_bwd = 3.5x fwd."""
    f = 4.0 * batch * nheads_q * seqlen_q * seqlen_k * head_dim
    if causal:
        f *= 0.5
    return {"fwd": f, "bwd": 2.5 * f, "fwd_bwd": 3.5 * f}[mode]


# ---------------------------------------------------------------------------
# correctness gate
# ---------------------------------------------------------------------------


def attention_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    softmax_scale: Optional[float] = None,
    upcast: bool = True,
) -> torch.Tensor:
    """Reference attention on CPU, (b, s, h, d) layout, fp32 (or in-dtype when
    ``upcast=False`` -- the "same math, test dtype" arm of the suite's error
    budget). Bottom-right-aligned causal, GQA by repeat_interleave. Mirrors
    flash_attn/cute/testing.py::attention_ref for the flags benchmarked here.
    """
    dt = torch.float32 if upcast else q.dtype
    qf, kf, vf = q.cpu().to(dt), k.cpu().to(dt), v.cpu().to(dt)
    b, sq, hq, d = qf.shape
    sk, hk = kf.shape[1], kf.shape[2]
    if hq != hk:
        kf = kf.repeat_interleave(hq // hk, dim=2)
        vf = vf.repeat_interleave(hq // hk, dim=2)
    scale = softmax_scale if softmax_scale is not None else d ** (-0.5)
    scores = torch.einsum("bthd,bshd->bhts", qf * scale, kf)
    if causal:
        mask = torch.triu(torch.ones(sq, sk, dtype=torch.bool), diagonal=sk - sq + 1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores.float(), dim=-1).to(dt)
    out = torch.einsum("bhts,bshd->bthd", attn, vf)
    return out


def check_against_oracle(
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    rtol: float = 2.0,
    atol: float = 1e-5,
) -> float:
    """The suite's error-budget assertion: candidate error may be at most
    ``rtol`` x the error of the same math done in the test dtype. Returns the
    ratio (candidate_err / in_dtype_err). Raises AssertionError on failure."""
    ref = attention_oracle(q, k, v, causal=causal, upcast=True)
    pt = attention_oracle(q, k, v, causal=causal, upcast=False)
    err = (out.cpu().float() - ref).abs().max().item()
    err_pt = (pt.float() - ref).abs().max().item()
    assert err <= rtol * err_pt + atol, (
        f"correctness gate FAILED: err={err:.3e} vs budget {rtol}*{err_pt:.3e}+{atol}"
    )
    return err / max(err_pt, 1e-30)


# ---------------------------------------------------------------------------
# result rows
# ---------------------------------------------------------------------------


@dataclass
class Row:
    candidate: str
    mode: str  # fwd | fwd_bwd
    batch: int
    seqlen: int
    nheads_q: int
    nheads_kv: int
    head_dim: int
    dtype: str
    causal: bool
    median_ms: float
    iqr_ms: float
    iters: int
    tflops: float
    tokens_per_s: float
    peak_mem_mb: float
    note: str = ""


class CsvWriter:
    def __init__(self, path: str):
        self.path = path
        self._wrote_header = False

    def write(self, row: Row) -> None:
        d = row.__dict__
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(d.keys()))
            if not self._wrote_header:
                w.writeheader()
                self._wrote_header = True
            w.writerow(d)


def make_qkv(
    batch, seqlen, hq, hk, hd, dtype, device="mps", requires_grad=False, seed=0
):
    torch.manual_seed(seed)
    q = torch.randn(batch, seqlen, hq, hd, device=device, dtype=dtype) * 0.5
    k = torch.randn(batch, seqlen, hk, hd, device=device, dtype=dtype) * 0.5
    v = torch.randn(batch, seqlen, hk, hd, device=device, dtype=dtype) * 0.5
    if requires_grad:
        for t in (q, k, v):
            t.requires_grad_()
    return q, k, v
