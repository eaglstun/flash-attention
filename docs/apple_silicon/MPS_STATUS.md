# FlashAttention on Apple Silicon (MPS) — Status

**Phase 1 complete (correctness) · Phase 3a complete (batched decode +
batched varlen).** This is the honesty document: what works, what is
degraded, what raises. An honest ❌ beats an optimistic ✅ that lies.

## What this is

A **second backend behind the frozen FlashAttention API**, written in plain
differentiable PyTorch on MPS. It is **not** a port of the CUDA kernels — CuTe
DSL has no Metal target, and `nvidia-cutlass-dsl` ships Linux-only wheels.
There is exactly **one implementation of the attention math**
(`flash_attn/mps/core.py`, fp32 accumulation everywhere, memory-bounded via
KV-chunked online-softmax forward and Q-chunked recompute backward), verified
against a CPU (and fp64) oracle by ~900 parity tests in `tests/mps/`. Both
public seams are thin adapters over that core:

| Seam                                    | Entry points                                                                                                            | Adapter                                                                                                                                         |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **FA2** — what third-party code imports | `from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_qkvpacked_func, …, flash_attn_with_kvcache` | `flash_attn/mps/fa2_backend.py` (implements the five-function `flash_attn_2_cuda` ABI: `fwd`, `bwd`, `varlen_fwd`, `varlen_bwd`, `fwd_kvcache`) |
| **FA4** — the modern API                | `from flash_attn.cute import flash_attn_func, flash_attn_varlen_func`                                                   | `flash_attn/mps/fa4_backend.py` (dispatch above the autograd.Function; autograd provides the backward)                                          |

**Correctness first; speed is Phase 2.** Nothing here is fused. If you need
CUDA-class throughput today, this backend is not it — what it gives you is
_correct numbers and correct gradients_ through the unmodified public API.

## The one deliberate behavioral divergence: masked-row `lse`

For rows whose keys are **all masked out** (e.g. causal with
`seqlen_q > seqlen_k`, or a narrow sliding window), the two generations of
CUDA kernels genuinely disagree, and each MPS seam matches _its own_
generation:

| API                                     | `lse` on fully-masked rows | Matches                                                           |
| --------------------------------------- | -------------------------- | ----------------------------------------------------------------- |
| FA2 (`flash_attn.flash_attn_interface`) | **`+inf`**                 | CUDA kernel, non-split path (`csrc/flash_attn/src/softmax.h:180`) |
| FA4 (`flash_attn.cute`)                 | **`-inf`**                 | CuTe kernel (`flash_attn/cute/softmax.py:225`)                    |

The core natively emits `-inf`; the FA2 adapter flips the sign. `out` is
exactly `0` on those rows in both seams, gradients are exactly `0`, and no NaN
appears anywhere, forward or backward. Pinned by
`tests/mps/test_fa2_seam.py::test_fa2_lse_masked_rows_plus_inf` and
`tests/mps/test_fa4_seam.py::test_fa4_lse_masked_rows_minus_inf`.

## Feature matrix — FA2 seam (`from flash_attn import …`)

✅ supported (parity-tested) · 🟡 degraded (correct but slow / caveats) · ❌ unsupported (raises `NotImplementedError`)

| Feature                                                                                          | Status | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| forward, fp16 / bf16 / fp32                                                                      | ✅     | fp32 accumulation always; error budget ≤ the suite's own 2× in-dtype-torch bound                                                                                                                                                                                                                                                                                                                                                                                                                 |
| causal (bottom-right aligned), MHA / GQA / MQA                                                   | ✅     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **backward / training**                                                                          | 🟡     | correct gradients, but implemented as **recompute + autograd**, not a fused kernel: the backward re-runs the forward per Q-chunk. Expect roughly 2× the forward work of a fused implementation. `dq/dk/dv` are filled in place per the C-extension ABI, including through the packed variants' `dqkv[:, :, i]` views                                                                                                                                                                             |
| sliding window / local attention (`window_size`)                                                 | ✅     | FA2 encoding (negative = infinite) translated at the seam                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| softcap                                                                                          | ✅     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ALiBi (`alibi_slopes`, `(h,)` or `(b, h)`)                                                       | ✅     | for causal ALiBi the CUDA kernel uses a row-shifted bias; output and gradients are identical, `lse` differs by a per-row constant                                                                                                                                                                                                                                                                                                                                                                |
| custom `softmax_scale`                                                                           | ✅     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| head dims: anything ≥ 1 that fits memory                                                         | ✅     | no multiple-of-8 kernel constraint (the interface still pads to 8 before calling; harmless)                                                                                                                                                                                                                                                                                                                                                                                                      |
| **varlen** (`flash_attn_varlen_func`, `…_qkvpacked`, `…_kvpacked`)                               | 🟡     | **batched (Phase 3a)**: pad-to-max + ONE per-batch-masked core call by default; falls back to the Phase-1 per-sequence loop when padding waste beats launch savings (extreme length skew) or when per-sequence SDPA launches win — measured heuristic in `flash_attn/mps/varlen.py`, sweep in `benchmarks/mps/bench_varlen.py --skew`. `seqused_k` supported. Still 🟡: whenever the ABI demands `lse` the forward is bound by the fp32 chunked core, not by launches — that ceiling is Phase 3b |
| `flash_attn_with_kvcache` (append + attend, `cache_seqlens`, `cache_batch_idx`, `cache_leftpad`) | 🟡     | in-place cache append matches CUDA (vectorized `index_put_`); attention is **one batched call** with per-batch `cache_seqlens`/`cache_leftpad` tail/head masking (Phase 3a — was a per-batch Python loop; 7.7x faster at decode shapes; ragged lengths are faster still). No backward (same as CUDA). Still 🟡: the mandatory `lse` keeps it on the fp32 chunked core, ~14x off the raw batched-SDPA bound (Phase 3b)                                                                            |
| `deterministic`, `num_splits`, `zero_tensors`                                                    | ✅     | accepted and **ignored** — perf/legacy knobs, not semantics. (The math here is deterministic anyway.)                                                                                                                                                                                                                                                                                                                                                                                            |
| **dropout (`dropout_p > 0`)**                                                                    | ❌     | raises. FA2's dropout mask comes from the CUDA Philox RNG _inside_ the kernel and is reproduced from `rng_state` in the backward. That RNG cannot be bit-matched on MPS, and a torch-side mask that can't be reproduced exactly in the recompute-based backward would produce **silently wrong gradients** — the exact failure mode this port exists to prevent. Set `dropout_p=0.0`                                                                                                             |
| `return_attn_probs` / `S_dmask`                                                                  | ❌     | raises (only meaningful with dropout; the S_dmask encoding is kernel-internal)                                                                                                                                                                                                                                                                                                                                                                                                                   |
| paged KV cache (`block_table`)                                                                   | ❌     | raises                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| rotary embedding inside `fwd_kvcache` (`rotary_cos/sin`)                                         | ❌     | raises — apply rotary to q/k _before_ calling. (The in-repo rotary oracle needs triton, which doesn't exist on macOS)                                                                                                                                                                                                                                                                                                                                                                            |
| `leftpad_k` in `varlen_fwd`                                                                      | ❌     | raises (not reachable from the public API; `cache_leftpad` in kvcache **is** supported)                                                                                                                                                                                                                                                                                                                                                                                                          |
| `torch.compile` capture of the `flash_attn::*` custom ops                                        | ❌     | on MPS the interface calls the Python implementations directly, bypassing `torch.ops` — the custom-op layer runs its backend impl below the Autograd dispatch key, which breaks the recompute-based backward. Eager semantics are identical                                                                                                                                                                                                                                                      |

## Feature matrix — FA4 seam (`from flash_attn.cute import …`)

| Feature                                                                                                                                                                                       | Status  | Notes                                                                             |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | --------------------------------------------------------------------------------- |
| forward + backward, fp16 / bf16 / fp32                                                                                                                                                        | ✅ / 🟡 | same core, same caveat: backward is recompute + autograd, not fused               |
| `(out, lse)` 2-tuple contract (also when `return_lse=False` → `(out, None)`)                                                                                                                  | ✅      |                                                                                   |
| causal, GQA/MQA, `window_size` (None = infinite; genuinely negative windows honored; the `left + right < 0` collapse quirk reproduced via the interface's own `_resolve_causal_local_window`) | ✅      |                                                                                   |
| softcap, `learnable_sink`, custom `softmax_scale`                                                                                                                                             | ✅      |                                                                                   |
| gradients through `lse` (dlse)                                                                                                                                                                | ✅      | the dispatch sits above the autograd.Function; `MPSFlashAttnFunc` propagates dlse |
| varlen: packed 3D + `cu_seqlens_{q,k}` (± `seqused_{q,k}` caps)                                                                                                                               | 🟡      | per-sequence Python loop — correct, slow. `lse` is `(nheads, total_q)`            |
| varlen: batched 4D + `seqused_{q,k}`                                                                                                                                                          | 🟡      | rows past `seqused_q` are zero-filled with `lse = -inf`                           |
| `num_splits`, `pack_gqa`, `deterministic`, `max_seqlen_*`, `min_seqlen_k`                                                                                                                     | ✅      | accepted and ignored (perf hints)                                                 |
| `score_mod` / `mask_mod` / `aux_tensors` / `aux_scalars`                                                                                                                                      | ❌      | raises — these are **cute-typed JIT callables**; they cannot run on torch tensors |
| block sparsity (`block_sparse_tensors*`)                                                                                                                                                      | ❌      | raises                                                                            |
| paged KV (`page_table`)                                                                                                                                                                       | ❌      | raises                                                                            |
| MLA (`qv`, hdim-512 absorption), `gather_kv_indices` / top-k                                                                                                                                  | ❌      | raises                                                                            |
| fp8                                                                                                                                                                                           | ❌      | not applicable on MPS (no fp8 torch support); dtype never reaches the backend     |

## Performance caveats, stated plainly

- **Nothing is fused.** The forward is chunked torch matmuls with an online
  softmax (memory-bounded, O(seqlen·chunk), so long sequences don't blow up
  unified memory) — but each chunk is a separate dispatch.
- **The backward recomputes.** Fused flash-attention also recomputes, but
  inside one kernel; here it is a Python loop over Q-chunks, each doing a
  forward + `torch.autograd.grad`. Training works and is exactly correct; it
  is not fast.
- **Varlen and kv-cache decode are batched (Phase 3a), not fused.** The
  varlen driver pads to the batch max and makes one per-batch-masked core
  call (a per-sequence loop is kept and auto-selected for pathological
  length skew — heuristic + measurements in `flash_attn/mps/varlen.py` and
  `docs/apple_silicon/BENCHMARKS.md`); `flash_attn_with_kvcache` is one
  vectorized append plus one batched masked attention call. Padded/masked
  positions contribute exactly zero, forward and backward (pinned by
  `tests/mps/test_batched_paths.py`).
- **The `lse` obligation is the remaining ceiling.** SDPA cannot return lse,
  so every FA2-seam forward (dense, varlen, kvcache) still runs the fp32
  chunked core — an order of magnitude off Apple's fused SDPA. That is
  Phase 3b (SDPA forward + separate chunked lse), not a batching problem.
- The **SDPA fast paths** (`F.scaled_dot_product_attention`) are used only
  where they are an _exact_ semantic match: plain/GQA attention with no lse
  requested — dense with `is_causal` only when `seqlen_q == seqlen_k`
  (SDPA's causal is top-left aligned; flash-attention's is bottom-right),
  varlen/ragged through an explicit boolean mask built to flash-attention's
  per-sequence bottom-right alignment (fully-masked rows are computed with a
  temporarily-unmasked row, then forced to exactly 0 with exactly-0
  gradients — never NaN).
- **Dropout RNG will never bit-match CUDA.** Currently moot because dropout
  raises, but if a future phase implements it, this line stays true.

## Verification (how we know)

- `tests/mps/test_attention_parity.py` — 790 core parity tests: CPU (fp32/fp64)
  oracle vs the core, all dtypes × causal × GQA × head_dim × seqlen ×
  window/softcap/ALiBi/sink, forward **and** backward, plus fully-masked-row
  NaN hunts.
- `tests/mps/test_fa2_seam.py`, `tests/mps/test_fa4_seam.py` — the seam
  contracts above, through the real public entry points.
- `tests/mps/test_batched_paths.py` (Phase 3a) — the ragged/skewed corners of
  the batched decode and varlen paths: wildly varying `cache_seqlens`
  (including 0) with in-place append, single-token decode, leftpad/batch_idx/
  window/ALiBi over ragged caches, batched-vs-looped varlen strategy parity
  (fwd + bwd), zero-length sequences, fully-masked rows through the masked-
  SDPA path (exact zeros, exact zero grads), and the skew heuristic itself.
- `tests/test_flash_attn.py` — the repo's own FA2 suite, run on MPS via
  `FLASH_ATTN_TEST_DEVICE=mps` (the suite was made device-parametrizable; on
  CUDA it is unchanged). **Phase-3a re-run: 14,180 passed** across
  varlen_causal / causal / varlen_output / output / kvcache / qkvpacked /
  varlen_qkvpacked / deterministic / varlen_deterministic / splitkv / bwd
  corner cases — every path the batching touches. Phase-1 run: a 1592-test
  dropout-free subset
  (output/varlen_output/causal/varlen_causal/kvcache/splitkv/deterministic/
  qkvpacked/bwd-corner-cases across seqlens 113–2048, d 64/128, fp16+bf16
  slice) — all green. Structural exclusions, stated plainly:
  - `dropout_p > 0` parametrizations raise `NotImplementedError` by design
    (verified explicitly — they fail with the documented error, not wrong
    numbers).
  - rotary kvcache cases can't run: their oracle imports
    `flash_attn.layers.rotary` → triton (Linux-only).
  - `test_flash_attn_race_condition` was run only at small seqlens
    ((1,239)/(239,1)/(97,97)/(128,128), d=64 — all passed): each case loops
    250 fwd+bwd iterations at batch 60 to shake out CUDA races, which takes
    hours on the unfused MPS backward at seqlen ≥ 512. The determinism
    property it checks held everywhere it ran.

## Not supported, not planned here

- Making `flash_attn/cute/` CuTe kernels compile for Metal (impossible).
- Bit-exact parity with CUDA kernels (different hardware, different reduction
  order; parity is defined by the error-budget assertions above).
- Speed. That is Phase 2 (benchmark torch-SDPA vs MLX vs
  `metal-flash-attention` before writing any Metal).
