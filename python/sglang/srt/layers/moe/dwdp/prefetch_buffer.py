"""Double-buffered async prefetch system for DWDP.

Uses CUDA IPC P2P reads (cudaMemcpyAsync over NVLink) to fetch peer expert
weights, with a dedicated CUDA stream and ping-pong buffers to overlap
buffer copies with MoE compute AND the attention phase of the next layer.

Key design (event-driven pipeline):
- 4 per-slot CUDA events (2 prefetch + 2 consume), following TRT-LLM #14453.
- No CPU-side synchronize() — all stream ordering is GPU-side via events.
- P2P copies on prefetch_stream overlap with attention + MoE on default stream.
- When IPC handles are cached (DwdpIpcHandleCache), per-layer open/close/sync
  is eliminated entirely.

Pipeline timeline:
  compute_stream:  [wait_prefetch[0]] [Attn(L)+MoE(L)] [wait_prefetch[1]] [Attn(L+1)+MoE(L+1)] ...
                                  |record consume[1]|                                |record consume[0]|
  copy_stream:     [P2P_L0] ────────────────────── [P2P_L1] ────────────────────── [P2P_L2] ...
                  |wait consume[0]|                  |wait consume[1]|
                  |record prefetch[0]|               |record prefetch[1]|
"""

from __future__ import annotations

import ctypes
import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.dwdp.dwdp_manager import (
        DwdpExpertLayout,
        DwdpIpcHandleCache,
        DwdpLayerHandleCollector,
    )

logger = logging.getLogger(__name__)


class DwdpPrefetchBuffer:
    """Double-buffered (ping-pong) async prefetch for DWDP expert weights.

    Buffer slot 0 is used by even-indexed MoE layers (0, 2, 4, ...),
    and slot 1 by odd-indexed (1, 3, 5, ...).

    Uses CUDA IPC P2P reads (cudaMemcpyAsync) to fetch peer expert weights
    over NVLink instead of NCCL all_gather. Each rank independently reads
    peer GPU memory — no collective synchronization is required.

    Event-driven pipeline (Fix 2): no CPU-side synchronize() when IPC
    handles are cached. Stream ordering is enforced via per-slot CUDA
    events following TRT-LLM #14453.
    """

    def __init__(
        self,
        layout: DwdpExpertLayout,
        num_moe_layers: int,
        param_full_shapes: Dict[str, torch.Size],
        param_full_strides: Dict[str, Tuple[int, ...]],
        param_dtypes: Dict[str, torch.dtype],
        device: torch.device,
        ipc_cache: Optional[DwdpIpcHandleCache] = None,
    ):
        self.layout = layout
        self.num_moe_layers = num_moe_layers
        self.param_full_shapes = param_full_shapes
        self.param_full_strides = param_full_strides
        self.param_dtypes = param_dtypes
        self.device = device
        self._ipc_cache = ipc_cache

        self.prefetch_stream = torch.cuda.Stream(device=device)

        num_prefetch_experts = layout.num_prefetch_experts
        num_experts_per_worker = layout.num_experts_per_worker
        dwdp_size = layout.dwdp_size
        dwdp_rank = layout.dwdp_rank

        # Per-param metadata. For each registered tensor we identify the
        # expert dimension as the logical dim whose size equals
        # ``num_experts_per_worker`` AND whose stride is maximal (i.e.
        # outermost in physical memory). This holds both for contiguous
        # tensors (expert dim 0) and for the MMA-layout SF strided views
        # (expert dim 5, but still outermost in physical memory).
        self.per_expert_bytes: Dict[str, int] = {}
        self.prefetch_view_shapes: Dict[str, torch.Size] = {}
        self.prefetch_view_strides: Dict[str, Tuple[int, ...]] = {}
        self.prefetch_view_dtypes: Dict[str, torch.dtype] = {}
        for name, shape in param_full_shapes.items():
            strides = param_full_strides[name]
            itemsize = param_dtypes[name].itemsize
            self.prefetch_view_dtypes[name] = param_dtypes[name]

            candidates = [i for i, s in enumerate(shape) if s == num_experts_per_worker]
            assert candidates, (
                f"No dim with size num_experts_per_worker={num_experts_per_worker} "
                f"for param {name} with shape {tuple(shape)}"
            )
            expert_dim = max(candidates, key=lambda i: strides[i])
            assert strides[expert_dim] == max(strides), (
                f"Expert dim {expert_dim} for param {name} is not the outermost "
                f"physical dim (shape={tuple(shape)}, strides={strides}). "
                f"Per-expert prefetch slicing requires experts to be outermost."
            )

            self.per_expert_bytes[name] = strides[expert_dim] * itemsize
            view_shape = list(shape)
            view_shape[expert_dim] = num_prefetch_experts
            self.prefetch_view_shapes[name] = torch.Size(view_shape)
            self.prefetch_view_strides[name] = tuple(strides)

        # Allocate 2 buffer slots (ping-pong)
        # buffers[buf_idx][param_name] = list of tensors, one per rank
        # Entry at dwdp_rank is None (local weights used directly)
        self.buffers: List[Dict[str, List[Optional[torch.Tensor]]]] = []
        for buf_idx in range(2):
            buffer = {}
            for param_name in param_full_shapes:
                tensor_list: List[Optional[torch.Tensor]] = [None] * dwdp_size
                for peer_rank in range(dwdp_size):
                    if peer_rank != dwdp_rank:
                        buf_bytes = num_prefetch_experts * self.per_expert_bytes[param_name]
                        tensor_list[peer_rank] = torch.empty(
                            buf_bytes,
                            dtype=torch.uint8,
                            device=device,
                        )
                buffer[param_name] = tensor_list
            self.buffers.append(buffer)

        # Per-slot CUDA events (4 total, following TRT-LLM #14453).
        # prefetch_events[buf_idx]: recorded on prefetch_stream after P2P copy completes.
        #   Default stream waits on this before reading (RAW hazard prevention).
        # consume_events[buf_idx]: recorded on default stream after wait_for_prefetch()
        #   for the OTHER slot. Fires after all prior default-stream work completes
        #   (including MoE kernel that read from the other slot). Prefetch stream
        #   waits on this before overwriting the slot (WAR hazard prevention).
        self.prefetch_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        self.consume_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

        logger.info(
            f"DwdpPrefetchBuffer allocated (event-driven pipeline): "
            f"num_prefetch_experts={num_prefetch_experts}, "
            f"per_expert_bytes={self.per_expert_bytes}, "
            f"ipc_cache={'enabled' if ipc_cache and ipc_cache.is_all_open else 'disabled'}, "
            f"params={list(param_full_shapes.keys())}"
        )

    def initialize_events(self) -> None:
        """Pre-record consume events so the first prefetch can proceed.

        Both consume_events are recorded as "already signaled" on the current
        stream, so the first prefetch_layer() calls won't deadlock waiting
        for a never-recorded event.
        """
        current_stream = torch.cuda.current_stream(self.device)
        self.consume_events[0].record(current_stream)
        self.consume_events[1].record(current_stream)

    def prefetch_layer(
        self,
        moe_layer_idx: int,
        peer_handles: Dict[int, Dict[str, Tuple[bytes, int]]],
    ) -> None:
        """Fetch peer expert weights via CUDA IPC P2P over NVLink.

        Event-driven pipeline: no CPU-side synchronize() when IPC handles
        are cached. P2P copies run on prefetch_stream and overlap with
        attention + MoE on the default stream. Stream ordering is enforced
        via per-slot CUDA events:
        - WAR: prefetch_stream waits on consume_events[buf_idx] before
          overwriting the slot (ensures prior compute is done).
        - RAW: prefetch_events[buf_idx] is recorded after P2P copy;
          default stream waits on it in wait_for_prefetch().

        When IPC handles are cached (DwdpIpcHandleCache with all handles open),
        this method uses cached mapped pointers directly (fast path).
        Otherwise, falls back to per-layer open/close/sync (slow path).
        """
        import os

        from sglang.srt.distributed.device_communicators.cuda_wrapper import (
            CudaRTLibrary,
            cudaIpcMemHandle_t,
        )

        _dwdp_debug = bool(os.environ.get("SGL_DWDP_DEBUG"))
        if _dwdp_debug:
            import torch.distributed as _dist
            _rank = _dist.get_rank() if _dist.is_initialized() else -1
            print(
                f"[DWDP_DEBUG] prefetch_layer begin rank={_rank} "
                f"moe_layer_idx={moe_layer_idx} "
                f"cached={self._ipc_cache is not None and self._ipc_cache.is_all_open}",
                flush=True,
            )

        cudart = CudaRTLibrary()
        buf_idx = moe_layer_idx % 2
        layout = self.layout
        dwdp_rank = layout.dwdp_rank
        num_prefetch_experts = layout.num_prefetch_experts

        use_cached = (
            self._ipc_cache is not None and self._ipc_cache.is_all_open
        )

        # --- P2P copy on prefetch stream (event-driven) ---
        with torch.cuda.stream(self.prefetch_stream):
            # WAR: wait for previous use of this buffer slot to finish.
            # consume_events[buf_idx] fires after all prior default-stream work
            # that read from this slot has completed.
            self.prefetch_stream.wait_event(self.consume_events[buf_idx])

            opened_ptrs = [] if not use_cached else None

            for peer_rank, handle_dict in peer_handles.items():
                src_expert_offset = layout.get_prefetch_src_offset(peer_rank)

                for param_name in self.per_expert_bytes:
                    if param_name not in handle_dict:
                        continue

                    expert_bytes = self.per_expert_bytes[param_name]
                    dst_tensor = self.buffers[buf_idx][param_name][peer_rank]
                    dst_ptr = dst_tensor.data_ptr()
                    data_size = num_prefetch_experts * expert_bytes

                    if use_cached:
                        # Fast path: use cached IPC pointer
                        src_ptr_int = self._ipc_cache.get_peer_ptr(
                            moe_layer_idx, peer_rank, param_name
                        )
                        src_ptr = src_ptr_int + src_expert_offset * expert_bytes
                    else:
                        # Slow path: open IPC handle on the fly
                        handle_bytes, offset = handle_dict[param_name]
                        handle = cudaIpcMemHandle_t()
                        handle.internal = (ctypes.c_byte * 128).from_buffer_copy(handle_bytes)
                        base_ptr = cudart.cudaIpcOpenMemHandle(handle)
                        base_ptr_int = int(base_ptr.value)
                        opened_ptrs.append(base_ptr_int)
                        src_ptr = base_ptr_int + offset + src_expert_offset * expert_bytes

                    cudart.cudaMemcpyAsync(
                        dst_ptr,
                        src_ptr,
                        data_size,
                        kind=4,  # cudaMemcpyDefault
                        stream=self.prefetch_stream.cuda_stream,
                    )

            # RAW: signal that prefetch for this slot is done.
            # Default stream will wait on this in wait_for_prefetch().
            self.prefetch_events[buf_idx].record(self.prefetch_stream)

        if use_cached:
            # Fast path: no sync needed, IPC handles stay open
            pass
        else:
            # Slow path: must synchronize before closing IPC handles.
            # This breaks the overlap but is required for correctness
            # when handles are not cached.
            cudart.cudaStreamSynchronize(self.prefetch_stream.cuda_stream)

            for ptr in opened_ptrs:
                cudart.cudaIpcCloseMemHandle(ptr)

        if _dwdp_debug:
            import torch.distributed as _dist
            _rank = _dist.get_rank() if _dist.is_initialized() else -1
            print(
                f"[DWDP_DEBUG] prefetch_layer end rank={_rank} "
                f"moe_layer_idx={moe_layer_idx}",
                flush=True,
            )

    def wait_for_prefetch(self, moe_layer_idx: int) -> None:
        """Default stream waits for prefetch of this layer to complete.

        Also records consume_events for the OTHER buffer slot, which provides
        the WAR signal for the next prefetch into that slot. This relies on
        CUDA stream in-order semantics: the consume event fires only after
        all prior work on the default stream (including the MoE kernel that
        reads from the other slot) has completed.
        """
        buf_idx = moe_layer_idx % 2
        other_buf = 1 - buf_idx
        current_stream = torch.cuda.current_stream(self.device)

        # RAW: wait for prefetch of this slot to complete
        current_stream.wait_event(self.prefetch_events[buf_idx])

        # WAR: record that the OTHER slot's data is being consumed.
        # This event fires after all prior default-stream work completes,
        # which includes the MoE kernel that read from other_buf.
        # The next prefetch into other_buf will wait on this event.
        self.consume_events[other_buf].record(current_stream)

    def get_buffer_views(
        self, moe_layer_idx: int
    ) -> Dict[str, List[Optional[torch.Tensor]]]:
        """Return buffer tensor views for the given MoE layer.

        Returns a dict mapping param_name -> list of tensors per rank.
        Entry at dwdp_rank is None (caller fills with local weight).
        Flat uint8 buffers are reinterpreted with the correct dtype and
        rebuilt with the same stride pattern as the original registered
        tensor (with ``num_prefetch_experts`` substituted for the expert
        dim). This preserves MMA-layout strided views over the physical
        storage laid out as ``(num_prefetch_experts, ...physical...)``.
        """
        buf_idx = moe_layer_idx % 2
        raw = self.buffers[buf_idx]
        views: Dict[str, List[Optional[torch.Tensor]]] = {}
        for param_name, tensor_list in raw.items():
            view_shape = self.prefetch_view_shapes[param_name]
            view_strides = self.prefetch_view_strides[param_name]
            view_dtype = self.prefetch_view_dtypes[param_name]
            view_list: List[Optional[torch.Tensor]] = []
            for t in tensor_list:
                if t is None:
                    view_list.append(None)
                else:
                    typed = t.view(view_dtype)
                    view_list.append(
                        torch.as_strided(
                            typed, view_shape, view_strides, storage_offset=0
                        )
                    )
            views[param_name] = view_list
        return views

    def cleanup(self) -> None:
        """Release prefetch buffers and events."""
        self.buffers.clear()
        self.prefetch_events.clear()
        self.consume_events.clear()
        self.prefetch_stream = None
