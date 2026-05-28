"""VMM Composite VA weight buffer for DWDP.

Uses CUDA Virtual Memory Management (VMM) to create a single virtual
address region backing a [num_routed_experts, ...] tensor per parameter.
Local expert weights are copied into their correct positions; remote
expert slots are filled by P2P prefetch. The MoE kernel sees one
contiguous tensor — no concatenation or multi-B kernel changes required.

Design: single VMM allocation per parameter per layer. This avoids
alignment gap issues that arise from multi-region layouts (VMM granularity
is 2MB, but expert data sizes are typically not 2MB-aligned).

No double-buffering is needed because each MoE layer is computed exactly
once per forward pass. The pool is filled by P2P prefetch, read by
compute, then the layer is done.

Adapted from TRT-LLM PR #14453 (VMM composite VA for DWDP).
Key difference: uses CU_MEM_HANDLE_TYPE_GENERIC instead of FABRIC
(Hopper compatibility), and IPC handles for cross-process P2P instead
of MNNVL fabric.
"""

from __future__ import annotations

import ctypes
import logging
import math
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# VMM granularity on Hopper (typically 2MB)
_VMM_GRANULARITY = 2 * 1024 * 1024


def _align_up(size: int, alignment: int) -> int:
    """Round up size to the nearest multiple of alignment."""
    return (size + alignment - 1) // alignment * alignment


class _CudaDriver:
    """Lazy-loaded CUDA Driver API wrapper for VMM operations."""

    def __init__(self):
        self._lib = None

    @property
    def lib(self):
        if self._lib is None:
            from cuda import cuda as cuda_driver
            self._lib = cuda_driver
        return self._lib

    def init(self):
        self.lib.cuInit(0)


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
        if err != 0:
            raise RuntimeError(f"cuMemMap failed at offset {offset}: err={err}")

    def unmap(self, offset: int, size: int) -> None:
        """Unmap a region of virtual address space."""
        va_ptr = _cuda.lib.CUdeviceptr(self.va + offset)
        err = _cuda.lib.cuMemUnmap(va_ptr, size)
        if err != 0:
            raise RuntimeError(f"cuMemUnmap failed at offset {offset}: err={err}")

    def set_access(self, device_id: int) -> None:
        """Set read/write access for a specific device on the entire region."""
        desc = _cuda.lib.CUmemAccessDesc()
        desc.location.type = _cuda.lib.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = device_id
        desc.flags = _cuda.lib.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        va_ptr = _cuda.lib.CUdeviceptr(self.va)
        err = _cuda.lib.cuMemSetAccess(va_ptr, self.size, [desc], 1)
        if err != 0:
            raise RuntimeError(f"cuMemSetAccess failed: err={err}")

    def free(self) -> None:
        """Free the virtual address reservation."""
        va_ptr = _cuda.lib.CUdeviceptr(self.va)
        err = _cuda.lib.cuMemAddressFree(va_ptr, self.size)
        if err != 0:
            logger.warning(f"cuMemAddressFree failed: err={err}")


class _VmmTensorView:
    """Wrapper that provides __cuda_array_interface__ for a VMM-backed pointer."""

    def __init__(self, ptr: int, shape: Tuple[int, ...], dtype: torch.dtype):
        self._ptr = ptr
        self._shape = shape
        self._dtype = dtype
        typestr_map = {
            torch.float32: "<f4",
            torch.float16: "<f2",
            torch.bfloat16: "<f2",
            torch.float8_e4m3fn: "|u1",
            torch.int32: "<i4",
            torch.uint8: "|u1",
        }
        self._typestr = typestr_map.get(dtype, "<f4")

    @property
    def __cuda_array_interface__(self):
        return {
            "data": (self._ptr, False),
            "shape": self._shape,
            "strides": None,
            "typestr": self._typestr,
            "version": 3,
        }


class DwdpVmmWeightBuffer:
    """VMM composite VA weight buffer for one MoE layer.

    Uses a single VMM allocation per parameter that covers the entire
    [num_routed_experts, ...] tensor. Local expert weights are copied
    into their correct positions at init time. Remote expert slots are
    filled by P2P prefetch at runtime.

    This design avoids alignment gap issues: since the entire tensor is
    one contiguous VMM allocation, expert i is always at byte offset
    i * per_expert_bytes, regardless of VMM granularity alignment.

    No double-buffering is needed because each MoE layer is computed
    exactly once per forward pass.
    """

    def __init__(
        self,
        layer_id: int,
        num_routed_experts: int,
        local_expert_start: int,
        local_expert_end: int,
        param_shapes: Dict[str, torch.Size],
        param_dtypes: Dict[str, torch.dtype],
        local_weights: Dict[str, torch.Tensor],
        device_id: int,
        dwdp_size: int,
    ):
        _cuda.init()

        self.layer_id = layer_id
        self.num_routed_experts = num_routed_experts
        self.local_expert_start = local_expert_start
        self.local_expert_end = local_expert_end
        self.num_local_experts = local_expert_end - local_expert_start
        self.device_id = device_id
        self.dwdp_size = dwdp_size

        # Per-param VMM state
        self._param_state: Dict[str, dict] = {}

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
            if err != 0:
                raise RuntimeError(
                    f"cuMemCreate failed for {name} on layer {layer_id}: err={err}"
                )

            # Reserve VA and map the handle
            err, va = _cuda.lib.cuMemAddressReserve(total_aligned, 0, 0, 0)
            if err != 0:
                raise RuntimeError(
                    f"cuMemAddressReserve failed for {name} on layer {layer_id}: err={err}"
                )
            va_int = int(va)

            region = VmmRegion(va_int, total_aligned)
            region.map(0, total_aligned, handle)
            region.set_access(device_id)

            # Copy local expert weights into their correct positions
            if name in local_weights:
                local_tensor = local_weights[name]
                local_byte_offset = local_expert_start * per_expert
                local_shape = local_tensor.shape

                # Create a view into the local expert region of the VMM allocation
                local_view = _VmmTensorView(
                    va_int + local_byte_offset, local_shape, dtype
                )
                local_vmm_tensor = torch.as_tensor(
                    local_view, device=f"cuda:{device_id}"
                )
                local_vmm_tensor.copy_(local_tensor)

            # Create tensor view over the entire composite VA
            full_shape = list(shape)
            full_shape[0] = num_routed_experts

            vmm_view = _VmmTensorView(va_int, tuple(full_shape), dtype)
            tensor = torch.as_tensor(vmm_view, device=f"cuda:{device_id}")

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
            f"DwdpVmmWeightBuffer created for layer {layer_id}: "
            f"num_routed_experts={num_routed_experts}, "
            f"local=[{local_expert_start},{local_expert_end}), "
            f"params={list(self._param_state.keys())}"
        )

    def get_tensor(self, param_name: str) -> torch.Tensor:
        """Return the composite VA tensor view for a parameter."""
        return self._param_state[param_name]["tensor"]

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
        logger.info(f"DwdpVmmWeightBuffer cleaned up for layer {self.layer_id}")
