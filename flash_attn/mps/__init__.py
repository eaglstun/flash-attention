"""Apple Silicon (MPS) backend for FlashAttention.

Phase 0 of the Apple Silicon port: this package makes ``import flash_attn`` work on
macOS by standing in for the CUDA extension ``flash_attn_2_cuda``. The actual
torch-based attention math lands in Phase 1. See docs/apple_silicon/PORT_PLAN.md.
"""
