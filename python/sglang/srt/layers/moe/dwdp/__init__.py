"""DWDP (Distributed Weight Data Parallelism) for MoE layers.

Tokens stay on-rank; expert weights are prefetched via NVLink P2P (CUDA IPC).
Weight assembly uses VMM Composite VA by default, mapping local + remote expert
weights into a single contiguous virtual address region — the MoE kernel sees
one tensor with no concatenation or multi-B kernel changes required.
Falls back to concat-based assembly if VMM is disabled.

This implementation works on non-Blackwell hardware (H800, A100, H20, etc.)
without requiring FlashInfer CuteDSL multi-B API.

Weight transfer uses cudaMemcpyAsync over NVLink (not NCCL all_gather),
enabling independent rank execution compatible with DP attention.

Ported from SGLang PR #23425 (NVIDIA/CUDA) and TRT-LLM PR #12136.
"""

from typing import Optional

_global_dwdp_manager = None


def get_global_dwdp_manager():
    """Return the global DwdpManager instance, or None if disabled."""
    return _global_dwdp_manager


def set_global_dwdp_manager(manager) -> None:
    """Set or clear the global DwdpManager singleton."""
    global _global_dwdp_manager
    _global_dwdp_manager = manager


def enable_dwdp() -> bool:
    """Return True if DWDP is active."""
    return _global_dwdp_manager is not None
