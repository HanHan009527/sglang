"""DWDP Manager — expert layout, IPC handle exchange, and weight assembly.

Supports two weight assembly modes:
1. VMM Composite VA (default): maps local + remote expert weights into a single
   contiguous virtual address region using ping-pong VMM slots. The MoE kernel
   sees one tensor — no concatenation or multi-B kernel changes required.
   Local expert weights are copied per-layer on the prefetch stream (overlapped
   with compute). Only 2 VMM slots (~1.48GB) instead of per-layer allocations (~44GB).
2. Concat fallback: concatenates local + prefetched weights into a new tensor
   each layer. Simpler but copies local experts unnecessarily.

Adapted from SGLang PR #23425 and TRT-LLM PR #12136.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Weight parameter names tracked per MoE layer.
# Includes both the FP8 weight tensors and their block-wise quantization scales.
# For BF16 models, the scale_inv parameters are absent and will be skipped.
WEIGHT_PARAM_NAMES = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale_inv",
    "w2_weight_scale_inv",
)


# ---------------------------------------------------------------------------
# Expert layout
# ---------------------------------------------------------------------------


@dataclass
class DwdpExpertLayout:
    """Defines expert-to-rank mapping for DWDP.

    Supports overlapping allocation: when ``num_experts_per_worker`` exceeds
    ``num_routed_experts // dwdp_size``, expert ranges across ranks overlap,
    reducing NVLink prefetch volume at the cost of extra local memory.
    """

    num_routed_experts: int
    dwdp_size: int
    dwdp_rank: int
    num_experts_per_worker: int  # experts stored locally per rank

    def __post_init__(self):
        assert self.num_experts_per_worker >= self.num_routed_experts // self.dwdp_size, (
            f"num_experts_per_worker ({self.num_experts_per_worker}) must be >= "
            f"num_routed_experts // dwdp_size ({self.num_routed_experts // self.dwdp_size})"
        )
        assert self.num_experts_per_worker <= self.num_routed_experts

    @property
    def num_prefetch_experts(self) -> int:
        """Number of experts to prefetch from each peer."""
        return math.ceil(
            (self.num_routed_experts - self.num_experts_per_worker)
            / (self.dwdp_size - 1)
        )

    @property
    def local_expert_start(self) -> int:
        return min(
            self.num_prefetch_experts * self.dwdp_rank,
            self.num_routed_experts - self.num_experts_per_worker,
        )

    @property
    def local_expert_end(self) -> int:
        return self.local_expert_start + self.num_experts_per_worker

    def peer_expert_range(self, peer_rank: int) -> Tuple[int, int]:
        """Return (start, end) of the expert range owned by *peer_rank*."""
        peer_start = min(
            self.num_prefetch_experts * peer_rank,
            self.num_routed_experts - self.num_experts_per_worker,
        )
        return peer_start, peer_start + self.num_experts_per_worker

    def get_prefetch_src_offset(self, peer_rank: int) -> int:
        """Offset (in number of experts) into peer's local tensor for prefetch."""
        peer_start, peer_end = self.peer_expert_range(peer_rank)
        if self.dwdp_rank < peer_rank:
            prefetch_start = peer_end - self.num_prefetch_experts
        else:
            prefetch_start = peer_start
        return prefetch_start - peer_start


# ---------------------------------------------------------------------------
# Per-layer IPC handle collector
# ---------------------------------------------------------------------------


class DwdpLayerHandleCollector:
    """Manages CUDA IPC handles for one MoE layer's weight tensors."""

    def __init__(self, layer_id: int):
        self.layer_id = layer_id
        self.local_weights: Dict[str, torch.Tensor] = {}
        self.peer_base_ptrs: Dict[Tuple[int, str], int] = {}
        self._ipc_mappings: List[int] = []  # base ptrs to close on cleanup

    def register(self, **kwargs: torch.Tensor) -> None:
        """Register local weight tensors for this layer.

        Clones each tensor to ensure it owns a separate CUDA allocation.
        This is required for IPC: cudaIpcGetMemHandle operates on the
        entire allocation, and if the tensor is a view (e.g. a slice of
        a larger tensor), the offset calculation via cuMemGetAddressRange
        would be wrong.  Cloning guarantees data_ptr() == allocation base.
        """
        for name in WEIGHT_PARAM_NAMES:
            if name in kwargs:
                self.local_weights[name] = kwargs[name].clone()

    def get_ipc_handles(self) -> Dict[str, Tuple[bytes, int]]:
        """Return (handle_bytes, offset) for each local weight tensor.

        Uses CudaRTLibrary (libcudart.so via ctypes) for IPC handle operations
        and libcuda.so.1 for cuMemGetAddressRange (CUDA Driver API) to compute
        the tensor's offset within its CUDA allocation.
        """
        import ctypes

        from sglang.srt.distributed.device_communicators.cuda_wrapper import (
            CudaRTLibrary,
        )

        lib = CudaRTLibrary()

        # Load CUDA driver API for cuMemGetAddressRange (not available in cudart)
        cu_lib = ctypes.CDLL("libcuda.so.1")

        # Initialize CUDA Driver API (required before any driver API calls)
        cu_init = cu_lib.cuInit
        cu_init.restype = ctypes.c_int
        cu_init.argtypes = [ctypes.c_uint]
        cu_init(0)

        # Must use the _v2 variant — the unversioned cuMemGetAddressRange
        # returns CUDA_ERROR_NOT_INITIALIZED (201) on modern drivers.
        cuMemGetAddressRange = cu_lib.cuMemGetAddressRange_v2
        cuMemGetAddressRange.restype = ctypes.c_int  # CUresult
        cuMemGetAddressRange.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),  # pbase
            ctypes.POINTER(ctypes.c_size_t),  # psize
            ctypes.c_size_t,                   # dptr
        ]

        handles = {}
        for name, tensor in self.local_weights.items():
            data_ptr = tensor.data_ptr()
            handle = lib.cudaIpcGetMemHandle(ctypes.c_void_p(data_ptr))

            # cudaIpcGetMemHandle returns a handle for the entire allocation.
            # cudaIpcOpenMemHandle returns the allocation base address.
            # We need offset = data_ptr - alloc_base to find the tensor within it.
            alloc_base = ctypes.c_size_t()
            alloc_size = ctypes.c_size_t()
            err = cuMemGetAddressRange(
                ctypes.byref(alloc_base), ctypes.byref(alloc_size), data_ptr
            )
            if err != 0:
                raise RuntimeError(f"cuMemGetAddressRange failed: error code {err}")
            offset = data_ptr - alloc_base.value
            handles[name] = (bytes(handle.internal), offset)
        return handles

    def open_peer_handles(
        self,
        all_handles: List[Dict[str, Tuple[bytes, int]]],
        dwdp_rank: int,
    ) -> None:
        """Open peer IPC handles and compute NVLink-accessible pointers.

        Uses CudaRTLibrary (libcudart.so via ctypes) instead of the
        nvidia-cuda-runtime-cu12 Python bindings for consistency with
        the rest of the SGLang codebase.
        """
        import ctypes

        from sglang.srt.distributed.device_communicators.cuda_wrapper import (
            CudaRTLibrary,
            cudaIpcMemHandle_t,
        )

        lib = CudaRTLibrary()

        for peer_rank, peer_handles in enumerate(all_handles):
            if peer_rank == dwdp_rank:
                continue
            for name, (handle_bytes, offset) in peer_handles.items():
                handle = cudaIpcMemHandle_t()
                handle.internal = (ctypes.c_byte * 128)(*handle_bytes)
                base_ptr = lib.cudaIpcOpenMemHandle(handle)
                base_ptr_int = base_ptr.value
                self._ipc_mappings.append(base_ptr_int)
                self.peer_base_ptrs[(peer_rank, name)] = base_ptr_int + offset

    def cleanup(self) -> None:
        """Close all IPC memory mappings."""
        import ctypes

        from sglang.srt.distributed.device_communicators.cuda_wrapper import (
            CudaRTLibrary,
        )

        lib = CudaRTLibrary()
        for base_ptr in self._ipc_mappings:
            try:
                lib.cudaIpcCloseMemHandle(ctypes.c_void_p(base_ptr))
            except RuntimeError:
                pass  # Already closed or invalid
        self._ipc_mappings.clear()
        self.peer_base_ptrs.clear()

    def close_peer_handles(self) -> None:
        """Close peer IPC handles (alias for cleanup, called after P2P copy)."""
        self.cleanup()


# ---------------------------------------------------------------------------
# DwdpManager — global singleton
# ---------------------------------------------------------------------------


class DwdpManager:
    """Orchestrates the DWDP lifecycle: weight registration, IPC exchange,
    prefetch buffer management, and weight assembly.

    Supports two weight assembly modes:
    - VMM Composite VA (default): maps local + remote expert weights into a
      single contiguous virtual address region using ping-pong VMM slots.
      No concatenation needed. Local expert copy per-layer on prefetch stream.
    - Concat fallback: concatenates local + prefetched weights into a new
      tensor each layer. Copies local experts unnecessarily.
    """

    def __init__(
        self,
        dwdp_size: int,
        dwdp_rank: int,
        num_routed_experts: int,
        num_moe_layers: int,
        first_k_dense_replace: int,
        total_num_layers: int,
        num_experts_per_worker: Optional[int] = None,
        use_vmm: bool = True,
    ):
        self.dwdp_size = dwdp_size
        self.dwdp_rank = dwdp_rank
        self.num_moe_layers = num_moe_layers
        self.first_k_dense_replace = first_k_dense_replace
        self.total_num_layers = total_num_layers
        self.use_vmm = use_vmm

        if num_experts_per_worker is None or num_experts_per_worker <= 0:
            num_experts_per_worker = num_routed_experts // dwdp_size

        self.layout = DwdpExpertLayout(
            num_routed_experts=num_routed_experts,
            dwdp_size=dwdp_size,
            dwdp_rank=dwdp_rank,
            num_experts_per_worker=num_experts_per_worker,
        )

        # Per-layer handle collectors, keyed by absolute layer_id
        self.layer_handles: Dict[int, DwdpLayerHandleCollector] = {}

        # Prefetch buffer (created after IPC exchange)
        self._prefetch_buffer = None

        # Track whether the initial prefetch has been triggered.
        # Must be triggered from forward_dwdp (not model_runner.forward_extend)
        # to avoid NCCL deadlock when DP ranks are desynchronized.
        self._initial_prefetch_done = False

        # VMM ping-pong slots (2 entries: slot 0 for even, slot 1 for odd).
        # Created in _init_vmm_buffers() after IPC exchange.
        self._vmm_slots: List["DwdpVmmWeightBuffer"] = []

        # Mapping from absolute layer_id to moe_layer_index (0-based)
        self._layer_id_to_moe_idx: Dict[int, int] = {}
        moe_idx = 0
        for layer_id in range(total_num_layers):
            if layer_id >= first_k_dense_replace:
                self._layer_id_to_moe_idx[layer_id] = moe_idx
                moe_idx += 1

        logger.info(
            f"DwdpManager initialized: dwdp_size={dwdp_size}, rank={dwdp_rank}, "
            f"num_routed_experts={num_routed_experts}, "
            f"num_experts_per_worker={self.layout.num_experts_per_worker}, "
            f"num_prefetch_experts={self.layout.num_prefetch_experts}, "
            f"local_expert_range=[{self.layout.local_expert_start}, {self.layout.local_expert_end}), "
            f"num_moe_layers={num_moe_layers}, first_k_dense={first_k_dense_replace}"
        )

    @property
    def expert_layout(self) -> DwdpExpertLayout:
        return self.layout

    # ----- Phase 2: Weight Registration -----

    def register_layer_weights(self, layer_id: int, **weight_tensors: torch.Tensor) -> None:
        """Called from process_weights_after_loading() for each MoE layer."""
        if layer_id not in self.layer_handles:
            self.layer_handles[layer_id] = DwdpLayerHandleCollector(layer_id)
        self.layer_handles[layer_id].register(**weight_tensors)

    # ----- Phase 3: IPC Exchange -----

    def exchange_ipc_handles(self) -> None:
        """AllGather IPC handles across DWDP group and enable P2P access."""
        import ctypes

        from sglang.srt.distributed.parallel_state import get_dwdp_group, get_dwdp_rank

        group = get_dwdp_group()
        self._all_handles: Dict[int, list] = {}

        for layer_id, collector in self.layer_handles.items():
            local_handles = collector.get_ipc_handles()
            all_handles = [None] * self.dwdp_size
            dist.all_gather_object(all_handles, local_handles, group=group.cpu_group)
            self._all_handles[layer_id] = all_handles

        # Enable P2P access between all GPU pairs in the DWDP group.
        # Required for cudaMemcpyAsync with cudaMemcpyDefault across devices.
        dwdp_rank = get_dwdp_rank()
        cudart = ctypes.CDLL("libcudart.so")
        for peer_rank in range(self.dwdp_size):
            if peer_rank == dwdp_rank:
                continue
            # cudaDeviceEnablePeerAccess(peerDevice, flags=0)
            # Returns cudaSuccess (0) or cudaErrorPeerAccessAlreadyEnabled (704)
            err = cudart.cudaDeviceEnablePeerAccess(peer_rank, 0)
            if err != 0 and err != 704:
                raise RuntimeError(
                    f"cudaDeviceEnablePeerAccess({peer_rank}) failed: error {err}"
                )

        logger.info(
            f"DWDP IPC handles exchanged for {len(self.layer_handles)} MoE layers, "
            f"P2P access enabled"
        )

    def init_prefetch_buffers(self) -> None:
        """Allocate double-buffered prefetch buffers and wire up IPC collectors.

        In VMM mode, also creates 2 ping-pong VMM composite VA slots.
        """
        from sglang.srt.layers.moe.dwdp.prefetch_buffer import DwdpPrefetchBuffer

        first_collector = next(iter(self.layer_handles.values()))
        param_shapes = {}
        param_dtypes = {}
        for name, tensor in first_collector.local_weights.items():
            param_shapes[name] = tensor.shape
            param_dtypes[name] = tensor.dtype

        self._prefetch_buffer = DwdpPrefetchBuffer(
            layout=self.layout,
            num_moe_layers=self.num_moe_layers,
            param_shapes=param_shapes,
            param_dtypes=param_dtypes,
            device=next(iter(first_collector.local_weights.values())).device,
            use_vmm=self.use_vmm,
        )

        # Re-key collectors and handles by moe_idx for prefetch buffer access.
        # layer_handles and _all_handles use absolute layer_id as keys,
        # but prefetch_layer() uses 0-based moe_layer_idx.
        collectors_by_moe_idx: Dict[int, DwdpLayerHandleCollector] = {}
        handles_by_moe_idx: Dict[int, list] = {}
        for layer_id, moe_idx in self._layer_id_to_moe_idx.items():
            if layer_id in self.layer_handles:
                collectors_by_moe_idx[moe_idx] = self.layer_handles[layer_id]
                handles_by_moe_idx[moe_idx] = self._all_handles[layer_id]

        self._prefetch_buffer.set_layer_collectors(
            collectors=collectors_by_moe_idx,
            all_handles_by_layer=handles_by_moe_idx,
        )

        # Create VMM composite VA buffers if enabled
        if self.use_vmm:
            self._init_vmm_buffers(collectors_by_moe_idx, param_shapes, param_dtypes)

        logger.info(
            f"DWDP prefetch buffers allocated (P2P mode, vmm={self.use_vmm})"
        )

    def _init_vmm_buffers(
        self,
        collectors_by_moe_idx: Dict[int, DwdpLayerHandleCollector],
        param_shapes: Dict[str, torch.Size],
        param_dtypes: Dict[str, torch.dtype],
    ) -> None:
        """Create 2 ping-pong VMM composite VA slots shared across all layers."""
        from sglang.srt.layers.moe.dwdp.vmm_buffer import DwdpVmmWeightBuffer

        device_id = torch.cuda.current_device()

        # Create 2 ping-pong VMM slots (shared across all layers)
        for slot_idx in range(2):
            slot = DwdpVmmWeightBuffer(
                slot_idx=slot_idx,
                num_routed_experts=self.layout.num_routed_experts,
                local_expert_start=self.layout.local_expert_start,
                local_expert_end=self.layout.local_expert_end,
                param_shapes=param_shapes,
                param_dtypes=param_dtypes,
                local_weights=None,  # No init-time copy; filled per-layer
                device_id=device_id,
                dwdp_size=self.dwdp_size,
            )
            self._vmm_slots.append(slot)

        # Wire VMM slots into the prefetch buffer for P2P destination routing
        self._prefetch_buffer.set_vmm_slots(self._vmm_slots)

        logger.info(f"VMM ping-pong slots created: 2 slots")

    def initialize_compute_events(self) -> None:
        """Pre-record initial compute events so the first prefetch can proceed."""
        assert self._prefetch_buffer is not None
        self._prefetch_buffer.initialize_compute_events()

    # ----- Phase 4: Forward Pass Operations -----

    def _get_peer_handles(self, layer_id: int) -> Dict[int, Dict[str, tuple]]:
        """Get peer IPC handle bytes for a layer, excluding self."""
        all_handles = self._all_handles[layer_id]
        peer_handles = {}
        for rank, handles in enumerate(all_handles):
            if rank != self.dwdp_rank:
                peer_handles[rank] = handles
        return peer_handles

    def prefetch_first_layers(self) -> None:
        """Async prefetch weights for the first 2 MoE layers."""
        assert self._prefetch_buffer is not None
        prefetch_stream = self._prefetch_buffer.prefetch_stream
        prefetch_stream.wait_stream(torch.cuda.current_stream(prefetch_stream.device))
        for moe_idx in range(min(2, self.num_moe_layers)):
            layer_id = self.first_k_dense_replace + moe_idx
            if layer_id in self.layer_handles:
                self._prefetch_buffer.prefetch_layer(
                    moe_layer_idx=moe_idx,
                    local_weights=self.layer_handles[layer_id].local_weights,
                )

    def get_assembled_weights(self, layer_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """Return full [num_routed_experts, ...] weight tensors for a MoE layer.

        In VMM mode, returns the composite VA tensor views from the ping-pong
        slot — no concatenation needed. Local experts are copied per-layer on
        the prefetch stream; remote experts are filled by P2P prefetch.

        In concat mode, waits for prefetch then assembles via torch.cat.
        """
        moe_idx = self._layer_id_to_moe_idx[layer_id]

        # VMM mode: return composite VA tensor views from ping-pong slot
        if self.use_vmm and self._vmm_slots:
            self._prefetch_buffer.wait_for_prefetch(moe_idx)
            slot = self._vmm_slots[moe_idx % 2]
            return {name: slot.get_tensor(name) for name in slot._param_state}

        # Concat fallback mode
        collector = self.layer_handles[layer_id]

        # Wait for prefetch to complete
        self._prefetch_buffer.wait_for_prefetch(moe_idx)

        # Get buffer views for prefetched peer weights
        buffer_views = self._prefetch_buffer.get_buffer_views(moe_idx)

        # Assemble: for each param, build [num_routed_experts, ...] by
        # placing each rank's experts in their global position.
        # Strategy: create output tensor, copy local + prefetched experts
        # into their correct global slots.
        assembled = {}
        layout = self.layout

        for param_name, local_tensor in collector.local_weights.items():
            num_experts_per_worker = layout.num_experts_per_worker
            expert_dim = 0  # For standard (non-MMA) layout, expert dim is 0

            # Create output tensor with full expert range
            full_shape = list(local_tensor.shape)
            full_shape[expert_dim] = layout.num_routed_experts
            full_tensor = torch.empty(
                full_shape, dtype=local_tensor.dtype, device=local_tensor.device
            )

            # Copy local experts into their global position
            local_start = layout.local_expert_start
            local_end = layout.local_expert_end
            full_tensor[local_start:local_end].copy_(local_tensor)

            # Copy prefetched peer experts into their global positions
            for peer_rank in range(layout.dwdp_size):
                if peer_rank == layout.dwdp_rank:
                    continue
                peer_start, peer_end = layout.peer_expert_range(peer_rank)
                peer_view = buffer_views[param_name][peer_rank]
                full_tensor[peer_start:peer_end].copy_(peer_view)

            assembled[param_name] = full_tensor

        return assembled

    def record_compute_and_prefetch_next(self, layer_id: int) -> None:
        """Record compute done event and trigger prefetch for layer_id + 2."""
        moe_idx = self._layer_id_to_moe_idx[layer_id]

        # Record compute done on default stream
        self._prefetch_buffer.record_compute_done(moe_idx)

        # Trigger prefetch for moe_idx + 2 (same buffer slot)
        next_moe_idx = moe_idx + 2
        if next_moe_idx < self.num_moe_layers:
            next_layer_id = self.first_k_dense_replace + next_moe_idx
            if next_layer_id in self.layer_handles:
                self._prefetch_buffer.prefetch_layer(
                    moe_layer_idx=next_moe_idx,
                    local_weights=self.layer_handles[next_layer_id].local_weights,
                )

    # ----- Phase 5: Cleanup -----

    def cleanup(self) -> None:
        """Release all DWDP resources."""
        if self._prefetch_buffer is not None:
            self._prefetch_buffer.cleanup()
            self._prefetch_buffer = None
        for slot in self._vmm_slots:
            slot.cleanup()
        self._vmm_slots.clear()
        for collector in self.layer_handles.values():
            collector.cleanup()
        self.layer_handles.clear()
        logger.info("DWDP resources cleaned up")
