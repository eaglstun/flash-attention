# Apple Silicon Port — Plan of Attack

**Status:** **Phase 0 ✅ · Phase 1 ✅ · Phase 2 ✅ (see `BENCHMARKS.md` — recommendation: torch-SDPA-based fast path, no MLX, no Metal)** · **Scope locked with Eric 2026-07-12** · **Fork:** `eaglstun/flash-attention`

> **Where we are (2026-07-12).** Attention works on Apple Silicon through _both_ public
> APIs, forward and backward, verified against a CPU/fp64 oracle. `from flash_attn import
flash_attn_func` — the import third-party code actually uses — runs on MPS. 888 parity
> tests green, plus the repo's **own** FA2 suite (1591 tests, dropout-free subset) passing
> on MPS. Feature matrix: `docs/apple_silicon/MPS_STATUS.md`. Nothing is fused yet; this is
> correct, not fast. See Phase 2.
>
> **Watch items carried forward:**
>
> - `float32/lse` passes its error budget at a ratio of **exactly 1.0000** (`atol=0`, zero
>   headroom). Not wrong, but brittle — if it ever goes red, look here first before
>   assuming a real regression.
> - On MPS the FA2 `_wrapped_*` aliases bind raw Python fns rather than `torch.ops.*`,
>   because `torch.library.custom_op` runs below the Autograd key and breaks the recompute
>   backward. Cost: **no `torch.compile` capture on MPS**. CUDA binding is unchanged.
> - `bwd`/`varlen_bwd` fill `dq/dk/dv` **in place** — the FA2 caller reads those tensors and
>   ignores all but `softmax_d` of the return. Break that and you get silently-zero grads.

## Scope (decided, do not re-litigate)

| Decision             | Choice                                                                                               |
| -------------------- | ---------------------------------------------------------------------------------------------------- |
| **Win condition**    | **Compat first, speed later.** Correct, importable, API-faithful attention on MPS. Perf is Phase 2+. |
| **Forward/backward** | **Both.** Eric intends to _train_ on this Mac, not just infer.                                       |
| **MLX**              | **Open question — benchmark it in Phase 2.** Not ruled in, not ruled out.                            |
| **Discipline**       | Correctness-first, **CPU as oracle**, parity harness before any kernel. Per Eric's MPS playbook.     |

Playbook: https://ai.ericeaglstun.com/deep-dives/porting-ml-to-apple-silicon/
(memory `apple-silicon-porting-deepdive`). The governing law of that playbook applies
double here: **MPS produces confidently-wrong numbers, not crashes.** "It ran" is not
"it's correct."

---

## The situation

This is not like the other Apple ports (finetrainers, cogkit, magicmotion, open-sora).
Those ride `torch.mps` and write no kernels. **This repo _is_ the kernel.**

Two facts that set the whole strategy:

1. **FA4 (`flash_attn/cute/`) cannot be installed on macOS, and no patch fixes it.**
   `nvidia-cutlass-dsl` transitively requires `nvidia-cutlass-dsl-libs-base`, which ships
   **manylinux-only** wheels (the native MLIR/NVVM compiler backend). CuTe DSL has no
   Metal target. 43 of ~55 files in `flash_attn/cute/` `import cutlass` at module scope.
   **Do not spend time trying to make `pip install flash_attn/cute` work on a Mac.**

2. **There is zero PyTorch compute anywhere in the FA4 path.** Not a slow path, not a
   debug path. `flash_fwd_combine.py`, `flash_bwd_preprocess.py`,
   `flash_bwd_postprocess.py` are all `@cute.jit`/`@cute.kernel` too. Everything JITs to
   PTX.

So this is **not a port. It is a second backend behind a frozen API.**

### The two seams

There are two independent interception points, and they are worth different things.

**Seam A — FA2 (`flash_attn/flash_attn_interface.py`). This is the one that unblocks the world.**
Third-party code does `from flash_attn import flash_attn_func` — the FA2 API — not
`flash_attn.cute`. And FA2 already abstracts its _entire_ compute backend behind a single
module bound to the name `flash_attn_gpu`, with exactly **five** functions:

```
flash_attn_gpu.fwd  ·  .bwd  ·  .varlen_fwd  ·  .varlen_bwd  ·  .fwd_kvcache
```

There is already an in-repo precedent for swapping it: the ROCm/Triton path at
`flash_attn_interface.py:20-23` rebinds `flash_attn_gpu` to a Triton module based on an
env var. **We do the same thing for MPS.** Implement those five functions in plain torch
and all ~1400 lines of the FA2 interface — the 7 public functions, the autograd Functions,
KV-cache, qkvpacked, varlen — light up unchanged.

The wall today: `import flash_attn_2_cuda` at `flash_attn_interface.py:23` is unguarded,
so `import flash_attn` hard-fails on macOS before anything else can happen.

**Seam B — FA4 (`flash_attn/cute/interface.py`). The modern API, nearly free to add.**
`flash_attn_func` (`interface.py:2723`) and `flash_attn_varlen_func` (`:2771`) are pure
passthroughs to `FlashAttnFunc.apply` / `FlashAttnVarlenFunc.apply` — the function bodies
contain **nothing** but the `.apply(...)` call. A device branch there sits **above** the
`autograd.Function`, so if the MPS forward is written in differentiable torch ops,
**autograd hands us the backward for free.** Contract to preserve: both always return a
**`(out, lse)` 2-tuple** (`return_lse=False` does not collapse it — `interface.py:2502`).

Both seams are thin adapters over **one shared core**. Build the core once.

### What we already have for free

- **A device-agnostic oracle.** `flash_attn/cute/testing.py::attention_ref` (line 326) is
  pure torch (`einsum` → softmax, upcasts to fp32 by default) and **runs on MPS
  unmodified.** Same file: `construct_local_mask` (:252), `generate_qkv` (:127). A
  near-duplicate lives at `tests/test_util.py` for the FA2 suite. This is the oracle —
  build the port test-first against it.
- **An arch-override precedent.** `FLASH_ATTENTION_ARCH` (`interface.py:88`) already
  bypasses `torch.cuda.get_device_capability()`.
- **Shallow device coupling.** Only ~25 `torch.cuda`/`.cuda()` hits across all of
  `flash_attn/cute/`, in 6 files (3 are benchmarks). In library code only **two** are
  genuinely CUDA-semantic: `get_device_capability()` (`interface.py:91`) and
  `get_device_properties().multi_processor_count` (`:566`, already `is_fake_mode()`-guarded).

---

## Phase 0 — Import & install survivability on macOS

**Goal:** `import flash_attn` and `from flash_attn.cute import flash_attn_func` both
succeed on an M-series Mac with zero NVIDIA packages installed. Calling into an
unimplemented path raises a **clear, actionable error** — never an `ImportError` on
`flash_attn_2_cuda`, never a segfault.

1. **Guard the FA2 C-extension import** (`flash_attn/flash_attn_interface.py:12-24`).
   Restructure the existing backend-selection block into an explicit chain:
   ROCm/Triton → CUDA (`flash_attn_2_cuda`) → **MPS** → raise. Select MPS when
   `torch.backends.mps.is_available()` and no CUDA extension is importable; allow an
   explicit override via `FLASH_ATTENTION_BACKEND={cuda,mps,triton}`.
2. **Platform-marker the CUDA deps out of `flash_attn/cute/pyproject.toml`** (lines 25-32)
   so a Mac install resolves. Environment markers on `nvidia-cutlass-dsl` and
   `quack-kernels`; keep `apache-tvm-ffi` and `torch-c-dlpack-ext` (both have real
   `macosx_*_arm64` wheels). Note the existing pin `nvidia-cutlass-dsl==4.6.0.dev0` does
   not exist on PyPI at all — it's a `pypi.nvidia.com` artifact. Leave that alone on Linux.
3. **Lazy-import the kernel modules in `flash_attn/cute/interface.py`** so the module-scope
   `import cutlass` wall is only hit when a CUDA path is actually taken.
4. **`_get_device_arch()` (`interface.py:76-92`) must not call `torch.cuda`** when there
   is no CUDA. Return an MPS sentinel.
5. Top-level `setup.py` already handles `darwin` (`get_platform()`, :77-89) and
   `FLASH_ATTENTION_SKIP_CUDA_BUILD` (:64) already skips `ext_modules`. Confirm, don't rebuild.

**Exit test:** on the M4, in a clean venv, `python -c "import flash_attn; import flash_attn.cute"` exits 0.

---

## Phase 1 — The MPS backend + the parity harness

**Goal:** correct forward **and** backward attention on MPS through both seams, verified
against CPU, with an honest feature matrix. This is the phase that delivers the actual win.

### 1a. The parity harness first (before any backend code)

Mirror the bitsandbytes precedent: the harness exists before the thing it tests.

- `tests/mps/` — device-parametrized. **Do not rewrite the oracle**; import
  `attention_ref` from `flash_attn/cute/testing.py`.
- Reuse the suite's existing error-budget assertion rather than inventing tolerances:
  `(out - out_ref).abs().max() <= rtol * (out_pt - out_ref).abs().max()`
  (`tests/cute/test_flash_attn.py:375`), where `out_ref` is fp32-upcast torch and `out_pt`
  is the same math in the test dtype. **CPU is the oracle; MPS is the defendant.**
- Blocker to route around: `tests/cute/conftest.py:11-29` shells out to `nvidia-smi`, and
  the test bodies hardcode `device="cuda"` (37 hits in `test_flash_attn.py`). Parametrize
  the device — don't fork the tests.
- **fp16/bf16 on MPS is the silent-wrongness zone.** Accumulate in fp32 everywhere.
  Every dtype gets a parity test, not just fp32.

### 1b. The shared core — `flash_attn/mps/core.py`

One differentiable forward, used by both seams:

```
_attention_forward(q, k, v, *, softmax_scale, causal, window_size, softcap,
                   learnable_sink, alibi_slopes, dropout_p, ...) -> (out, lse)
```

- **Fast path:** `F.scaled_dot_product_attention` when the flags allow (plain / causal /
  GQA / dropout-free). It is differentiable, it is the memory-efficient path on MPS, and
  it is what MPS is actually good at.
- **Manual path:** explicit masked matmul + fp32 online softmax for the flags SDPA can't
  express (softcap, local/sliding window, learnable sink, ALiBi). **Chunk over K blocks**
  so we don't materialize the full `seqlen²` score matrix in unified memory — that's
  "flash" in the memory sense even with no custom kernel.
- **`lse` must be produced differentiably** (`logsumexp`), not reconstructed after the fact
  — Seam B's free-backward depends on it.

### 1c. Backward

The two seams need backward differently, and this is the one real subtlety:

- **Seam B (FA4)** gets it free: differentiable forward above `autograd.Function` → autograd.
- **Seam A (FA2)** calls `flash_attn_gpu.bwd(...)` _explicitly_ from inside its own
  `autograd.Function.backward`, so we must supply a real `bwd`. Implement it as
  **recompute + `torch.autograd.grad`**: re-run `_attention_forward` under `enable_grad`
  on the saved `q,k,v` and differentiate. Flash-attention recomputes in the backward
  anyway, so this is philosophically honest and exactly correct — it just isn't fused.
  Same core function, no second implementation of the math to keep in sync.

### 1d. The five-function FA2 adapter — `flash_attn/mps/fa2_backend.py`

Mimic the `flash_attn_2_cuda` ABI: `fwd`, `bwd`, `varlen_fwd`, `varlen_bwd`,
`fwd_kvcache`. Match the C-extension's return shapes exactly (`out, softmax_lse, S_dmask,
rng_state` etc.) — the FA2 interface unpacks them positionally. Varlen: unpad/pad around
the dense path (correct first; fast never mind).

### 1e. `docs/apple_silicon/MPS_STATUS.md` — the honesty document

Same as the bitsandbytes README table. Every feature is **supported / degraded / unsupported**,
and unsupported means a **clear `NotImplementedError`**, never a wrong number.

| Feature                                                     | Expected Phase-1 status                                                                      |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| causal, GQA/MQA, fp16/bf16/fp32                             | supported (SDPA fast path)                                                                   |
| softcap, local/sliding window, ALiBi, sink                  | supported (manual chunked path, slower)                                                      |
| varlen                                                      | supported (unpad/pad — correct, not fast)                                                    |
| KV-cache (`fwd_kvcache`)                                    | supported                                                                                    |
| backward / training                                         | supported (recompute + autograd — correct, not fused)                                        |
| `num_splits`, `pack_gqa`, `deterministic`                   | **accepted and ignored** (perf knobs, not semantics) — don't raise                           |
| dropout (`dropout_p > 0`)                                   | supported, but **RNG will not bit-match CUDA** — document loudly                             |
| `return_attn_probs` / `S_dmask`                             | `NotImplementedError`                                                                        |
| `score_mod` / `mask_mod`                                    | `NotImplementedError` — these are **cute-typed callables**, they do not run on torch tensors |
| block-sparse, paged KV / `page_table`, `num_splits` combine | `NotImplementedError`                                                                        |
| fp8, MLA (`qv` 512 path), `gather_kv_indices`, topk         | `NotImplementedError`                                                                        |

**Exit test:** `pytest tests/mps/` green across dtypes × causal × GQA × head_dim × varlen,
forward and backward, against the CPU oracle. A real model trains a few steps on the M4.

---

## Phase 2 — Benchmark before committing to a fast path

Only now, with a correct baseline and a working harness, do we spend money on speed.
Measure on the shapes Eric actually cares about (head_dim 64/128, the seqlens his training
runs use):

1. **torch SDPA on MPS** (the Phase-1 baseline).
2. **MLX's SDPA / attention kernels** — real fused Metal, already written. Cost is a
   dependency + torch-MPS ↔ MLX tensor interop (both unified memory; interop is fiddly,
   not impossible). Potentially most of flash-attention's win for a fraction of the MSL.
3. **Philip Turner's `metal-flash-attention`** — genuine prior art, hand-tuned MSL, has
   **forward and backward**. Already linked from this repo (`usage.md:126`). Read it before
   writing a single line of Metal.

Deliverable: a numbers table and a recommendation. **The MLX question gets answered here, not before.**

> **Done (2026-07-12).** `docs/apple_silicon/BENCHMARKS.md` has the numbers,
> methodology, and the Phase 3 recommendation: build on torch MPS SDPA
> (SDPA fwd + chunked lse, SDPA-powered chunked bwd, batched varlen/decode);
> MLX rejected (fwd merely ties SDPA, bwd unfused O(L²) and bf16 grads fail
> the error budget); MFA upstream unusable (single-head, no causal, Swift-only,
> dormant). Benchmark harness lives in `benchmarks/mps/` and re-runs.
> Landed here: `_Q_CHUNK_SIZE_BWD` 256→512 (4-9% bwd win), `enable_gqa` in
> the SDPA fast path.

## Phase 3 — The fast path

Whichever Phase 2 picks. If it's hand-written MSL, it's the bitsandbytes shape (own
`.metal`, own `.metallib`, packaged via package-data glob so it survives `pip install`) and
the backward pass is where it gets genuinely hard — dQ/dK/dV recompute, atomics,
determinism. Scope that phase when we get there, not now.

---

## Workflow

Mirrors finetrainers / bitsandbytes:

- This document is the executable spec. Update it as reality argues back.
- Work in a **feature worktree**, not `main`.
- One phase per dispatch; review before the next phase starts.
- `agent_space/` is the disposable scratch dir (per repo `CLAUDE.md`).
- Ruff before committing: `ruff check flash_attn/ --fix && ruff format flash_attn/`
  (note: the big kernel files are excluded from auto-format — leave them alone).

## Non-goals

- Making `flash_attn/cute/` (CuTeDSL) compile on Apple. Impossible. Stop.
- Matching CUDA bit-for-bit on dropout RNG.
- Upstreaming to Dao-AILab. This is Eric's fork.
- Speed, in Phase 0 and 1. **Correctness has right of way.**
