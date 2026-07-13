"""Apple Silicon (MPS) backend for FlashAttention.

One shared, parity-tested attention core (``core.py``: fp32 accumulation,
KV-chunked online-softmax forward, Q-chunked recompute backward) behind two
thin seam adapters:

- ``fa2_backend.py`` — the five-function ``flash_attn_2_cuda`` ABI
  (``fwd`` / ``bwd`` / ``varlen_fwd`` / ``varlen_bwd`` / ``fwd_kvcache``),
  selected by ``flash_attn/flash_attn_interface.py`` when MPS is the device.
- ``fa4_backend.py`` — the ``flash_attn.cute`` public API twins, dispatched
  from ``flash_attn/cute/interface.py`` above its autograd.Function so
  autograd provides the backward.
- ``varlen.py`` — the differentiable per-sequence varlen driver both share.

Feature matrix, deliberate divergences (masked-row ``lse`` sign per API
generation) and performance caveats: docs/apple_silicon/MPS_STATUS.md.
Plan and history: docs/apple_silicon/PORT_PLAN.md.
"""
