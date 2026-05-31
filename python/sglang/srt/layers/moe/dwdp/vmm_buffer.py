"""VMM Composite VA weight buffer for DWDP.

Uses CUDA Virtual Memory Management (VMM) to create a single virtual
address region backing a [num_routed_experts, ...] tensor per parameter.
Remote expert slots are filled by P2P prefetch; local expert weights are
copied into their correct positions per-layer on the prefetch stream.
The MoE kernel sees one contiguous tensor — no concatenation or multi-B
kernel changes required.

Design: single VMM allocation per parameter per slot. This avoids
alignment gap issues that arise from multi-region layouts (VMM granularity
is 2MB, but expert data sizes are typically not 2MB-aligned).

Ping-pong slot model: only 2 VMM slots exist (slot 0 for even MoE layers,
slot 1 for odd). Each slot is reused across layers. Before each layer's
P2P prefetch, local expert weights are copied into the slot on the
prefetch stream (overlapping with previous layer's compute). This reduces
persistent memory from ~44GB (60 per-layer allocations) to ~1.48GB (2 slots).

Adapted from TRT-LLM PR #14453 (VMM composite VA for DWDP).
Key difference: uses CU_MEM_HANDLE_TYPE_GENERIC instead of FABRIC
(Hopper compatibility), and IPC handles for cross-process P2P instead
of MNNVL fabric.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# VMM granularity on Hopper (typically 2MB)
_VMM_GRANULARITY = 2 * 1024 * 1024


def _align_up(size: int, alignment: int) -> int:
    """Round up size to the nearest multiple of alignment."""
    return (size + alignment - 1) // alignment * alignment


class _CudaDriver:
    """Lazy-loaded CUDA Driver API wrapper for VMM operations.

    Uses cuda.bindings.driver (cuda-python 13.x API).
    In this API, functions return (CUresult, ...) tuples; void-like
    functions return (CUresult,). We normalize error checking via _check().
    """

    def __init__(self):
        self._lib = None
        self._initialized = False

    @property
    def lib(self):
        if self._lib is None:
            from cuda.bindings import driver as cuda_driver
            self._lib = cuda_driver
        return self._lib

    def init(self):
        if self._initialized:
            return
        self.lib.cuInit(0)
        self._initialized = True

    @staticmethod
    def check(err, msg=""):
        """Check CUDA Driver API return value. Handles tuple returns from cuda-python 13.x."""
        if isinstance(err, tuple):
            err = err[0]
        if isinstance(err, int) and err != 0:
            raise RuntimeError(f"{msg}: err={err}")
        if hasattr(err, 'value') and err.value != 0:
            raise RuntimeError(f"{msg}: err={err}")


# Module-level singleton
_cuda = _CudaDriver()


class VmmRegion:
    """Manages a single VMM virtual address region with mapped handles."""

    def __init__(self, va: int, size: int):
        self.va = va
        self.size = size

    def map(self, offset: int, size: int, handle) -> None:
        """Map a physical memory handle at a given offset in the VA region."""
        va_ptr = _cuda.lib.CUdeviceptr(self.va + offset)
        err = _cuda.lib.cuMemMap(va_ptr, size, 0, handle, 0)
        _cuda.check(err, f"cuMemMap failed at offset {offset}")

    def unmap(self, offset: int, size: int) -> None:
        """Unmap a region of virtual address space."""
        va_ptr = _cuda.lib.CUdeviceptr(self.va + offset)
        err = _cuda.lib.cuMemUnmap(va_ptr, size)
        _cuda.check(err, f"cuMemUnmap failed at offset {offset}")

    def set_access(self, device_id: int) -> None:
        """Set read/write access for a specific device on the entire region."""
        desc = _cuda.lib.CUmemAccessDesc()
        desc.location.type = _cuda.lib.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = device_id
        desc.flags = _cuda.lib.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        va_ptr = _cuda.lib.CUdeviceptr(self.va)
        err = _cuda.lib.cuMemSetAccess(va_ptr, self.size, [desc], 1)
        _cuda.check(err, "cuMemSetAccess failed")

    def free(self) -> None:
        """Free the virtual address reservation."""
        va_ptr = _cuda.lib.CUdeviceptr(self.va)
        err = _cuda.lib.cuMemAddressFree(va_ptr, self.size)
        _cuda.check(err, "cuMemAddressFree failed")


class _VmmTensorView:
    """Wrapper that provides __cuda_array_interface__ for a VMM-backed pointer.

    Uses a byte-level typestr ("|u1") to create a uint8 tensor from the
    raw VMM pointer, then the caller applies .view(dtype) to reinterpret
    the bytes with the correct dtype.  This avoids the __cuda_array_interface__
    typestr ambiguity problem: float8_e4m3fn and uint8 both map to "|u1",
    and bfloat16 and float16 both map to "<f2".
    """

    def __init__(self, ptr: int, shape: Tuple[int, ...], dtype: torch.dtype):
        self._ptr = ptr
        self._shape = shape
        self._dtype = dtype
        # Always use byte-level typestr; caller must .view(dtype)
        self._typestr = "|u1"

    @property
    def __cuda_array_interface__(self):
        # Compute byte-level shape: total elements * itemsize
        numel = 1
        for s in self._shape:
            numel *= s
        byte_shape = (numel * self._dtype.itemsize,)
        return {
            "data": (self._ptr, False),
            "shape": byte_shape,
            "strides": None,
            "typestr": self._typestr,
            "version": 3,
        }


def _vmm_tensor_from_ptr(ptr: int, shape: Tuple[int, ...], dtype: torch.dtype, device_id: int) -> torch.Tensor:
    """Create a tensor from a VMM pointer with correct dtype and shape.

    Uses _VmmTensorView (byte-level __cuda_array_interface__) then
    .view(dtype).reshape(shape) to get the right dtype and shape without
    typestr ambiguity.
    """
    view = _VmmTensorView(ptr, shape, dtype)
    byte_tensor = torch.as_tensor(view, device=f"cuda:{device_id}")
    return byte_tensor.view(dtype).reshape(shape)


class DwdpVmmWeightBuffer:
    """VMM composite VA weight buffer for a ping-pong slot.

    Uses a single VMM allocation per parameter that covers the entire
    [num_routed_experts, ...] tensor. Local expert weights are copied
    into their correct positions per-layer via copy_local_experts().
    Remote expert slots are filled by P2P prefetch at runtime.

    This design avoids alignment gap issues: since the entire tensor is
    one contiguous VMM allocation, expert i is always at byte offset
    i * per_expert_bytes, regardless of VMM granularity alignment.

    Ping-pong slot model: only 2 slots exist (slot 0 for even MoE layers,
    slot 1 for odd). Each slot is reused across layers. Before each layer's
    P2P prefetch, local expert weights are copied into the slot on the
    prefetch stream (overlapping with previous layer's compute).
    """

    def __init__(
        self,
        slot_idx: int,
        num_routed_experts: int,
        local_expert_start: int,
        local_expert_end: int,
        param_shapes: Dict[str, torch.Size],
        param_dtypes: Dict[str, torch.dtype],
        local_weights: Optional[Dict[str, torch.Tensor]] = None,
        device_id: int = 0,
        dwdp_size: int = 1,
    ):
        _cuda.init()

        self.slot_idx = slot_idx
        self.num_routed_experts = num_routed_experts
        self.local_expert_start = local_expert_start
        self.local_expert_end = local_expert_end
        self.num_local_experts = local_expert_end - local_expert_start
        self.device_id = device_id
        self.dwdp_size = dwdp_size

        # Per-param VMM state
        self._param_state: Dict[str, dict] = {}

        # Pre-created views into the local expert region of each param.
        # Used by copy_local_experts() to copy per-layer local weights.
        self._local_views: Dict[str, torch.Tensor] = {}

        # Compute per-expert byte sizes
        self.per_expert_bytes: Dict[str, int] = {}
        for name, shape in param_shapes.items():
            itemsize = param_dtypes[name].itemsize
            self.per_expert_bytes[name] = shape[1:].numel() * itemsize if len(shape) > 1 else itemsize

        # Create VMM layout for each parameter
        for name, shape in param_shapes.items():
            dtype = param_dtypes[name]
            itemsize = dtype.itemsize
            expert_dim_size = shape[1:].numel() if len(shape) > 1 else 1
            per_expert = expert_dim_size * itemsize

            # Total size for all experts
            total_bytes = num_routed_experts * per_expert
            total_aligned = _align_up(total_bytes, _VMM_GRANULARITY)

            if total_aligned == 0:
                continue

            # Create single VMM allocation for the entire tensor
            prop = _cuda.lib.CUmemAllocationProp()
            prop.type = _cuda.lib.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            prop.requestedHandleTypes = _cuda.lib.CUmemHandleType.CU_MEM_HANDLE_TYPE_GENERIC
            prop.location.type = _cuda.lib.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            prop.location.id = device_id

            err, handle = _cuda.lib.cuMemCreate(total_aligned, prop, 0)
            _cuda.check(err, f"cuMemCreate failed for {name} on slot {slot_idx}")

            # Reserve VA and map the handle
            err, va = _cuda.lib.cuMemAddressReserve(total_aligned, 0, 0, 0)
            _cuda.check(err, f"cuMemAddressReserve failed for {name} on slot {slot_idx}")
            va_int = int(va)

            region = VmmRegion(va_int, total_aligned)
            region.map(0, total_aligned, handle)
            region.set_access(device_id)

            # Pre-create view into the local expert region for copy_local_experts()
            local_byte_offset = local_expert_start * per_expert
            num_local = local_expert_end - local_expert_start
            local_shape = list(shape)
            local_shape[0] = num_local
            local_view = _VmmTensorView(
                va_int + local_byte_offset, tuple(local_shape), dtype
            )
            self._local_views[name] = _vmm_tensor_from_ptr(
                va_int + local_byte_offset, tuple(local_shape), dtype, device_id
            )

            # Copy local expert weights at init time if provided (backward compat)
            if local_weights is not None and name in local_weights:
                self._local_views[name].copy_(local_weights[name])

            # Create tensor view over the entire composite VA
            full_shape = list(shape)
            full_shape[0] = num_routed_experts

            vmm_view = _VmmTensorView(va_int, tuple(full_shape), dtype)
            tensor = _vmm_tensor_from_ptr(va_int, tuple(full_shape), dtype, device_id)

            self._param_state[name] = {
                "va_int": va_int,
                "total_aligned": total_aligned,
                "total_bytes": total_bytes,
                "handle": handle,
                "region": region,
                "tensor": tensor,
                "full_shape": tuple(full_shape),
                "dtype": dtype,
            }

        logger.info(
            f"DwdpVmmWeightBuffer created for slot {slot_idx}: "
            f"num_routed_experts={num_routed_experts}, "
            f"local=[{local_expert_start},{local_expert_end}), "
            f"params={list(self._param_state.keys())}"
        )

    def get_tensor(self, param_name: str) -> torch.Tensor:
        """Return the composite VA tensor view for a parameter."""
        return self._param_state[param_name]["tensor"]

    def copy_local_experts(self, local_weights: Dict[str, torch.Tensor]) -> None:
        """Copy local expert weights into their correct positions in the VMM slot.

        Must be called on the prefetch stream so the copy overlaps with
        previous layer's compute.

        ``local_weights`` contains only this rank's local experts, i.e.
        shape [num_local_experts, ...] (already sliced by EP).  The VMM
        _local_views are positioned at the correct offset within the
        composite VA, so we copy the entire tensor directly.
        """
        for name, local_tensor in local_weights.items():
            if name in self._local_views:
                self._local_views[name].copy_(local_tensor)

    def get_pool_ptr(self, param_name: str, peer_rank: int, layout) -> Tuple[int, int]:
        """Return (dst_ptr, copy_bytes) for P2P prefetch into the VMM allocation.

        Since the entire tensor is a single contiguous VMM allocation,
        the destination pointer is simply:
            va_int + peer_expert_start * per_expert_bytes
        """
        state = self._param_state[param_name]
        va_int = state["va_int"]
        per_expert = self.per_expert_bytes[param_name]

        peer_start, peer_end = layout.peer_expert_range(peer_rank)

        dst_ptr = va_int + peer_start * per_expert
        copy_bytes = (peer_end - peer_start) * per_expert

        return dst_ptr, copy_bytes

    def cleanup(self) -> None:
        """Release all VMM resources."""
        for name, state in self._param_state.items():
            region = state["region"]
            total_aligned = state["total_aligned"]

            # Unmap and free VA
            region.unmap(0, total_aligned)
            region.free()

            # Release handle
            _cuda.lib.cuMemRelease(state["handle"])

        self._param_state.clear()
        self._local_views.clear()
        logger.info(f"DwdpVmmWeightBuffer cleaned up for slot {self.slot_idx}")
