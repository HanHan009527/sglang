"""Double-buffered async prefetch system for DWDP.

Uses CUDA IPC P2P reads (cudaMemcpyAsync over NVLink) to fetch peer expert
weights, with a dedicated CUDA stream and ping-pong buffers to overlap
buffer copies with MoE compute AND the attention phase of the next layer.

Key design (Fix 2 — event-driven pipeline):
- 4 per-slot CUDA events (2 prefetch + 2 consume), following TRT-LLM #14453.
- No CPU-side synchronize() — all stream ordering is GPU-side via events.
- P2P copies on prefetch_stream overlap with attention + MoE on default stream.

Pipeline timeline:
  compute_stream:  [wait_prefetch[0]] [Attn(L)+MoE(L)] [wait_prefetch[1]] [Attn(L+1)+MoE(L+1)] ...
                                  |record consume[1]|                                |record consume[0]|
  copy_stream:     [P2P_L0] ────────────────────── [P2P_L1] ────────────────────── [P2P_L2] ...
                  |wait consume[0]|                  |wait consume[1]|
                  |record prefetch[0]|               |record prefetch[1]|

In VMM mode, P2P copies write directly into VMM ping-pong slots instead
of separate concat buffer tensors. Local expert weights are also copied
into the VMM slot on the prefetch stream before P2P copies begin.
"""

from __future__ import annotations

import ctypes
import json
import logging
import math
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.dwdp.dwdp_manager import DwdpExpertLayout, DwdpIpcHandleCache, DwdpLayerHandleCollector
    from sglang.srt.layers.moe.dwdp.vmm_buffer import DwdpVmmWeightBuffer

logger = logging.getLogger(__name__)


class DwdpPrefetchBuffer:
    """Double-buffered (ping-pong) async prefetch for DWDP expert weights.

    Buffer slot 0 is used by even-indexed MoE layers (0, 2, 4, ...),
    and slot 1 by odd-indexed (1, 3, 5, ...).

    Uses CUDA IPC P2P reads (cudaMemcpyAsync) to fetch peer expert weights
    over NVLink instead of NCCL all_gather. Each rank independently reads
    peer GPU memory — no collective synchronization is required.

    In VMM mode, concat buffers are not allocated — P2P copies write
    directly into VMM ping-pong slots, and local expert weights are
    copied into the slot on the prefetch stream.
    """

    def __init__(
        self,
        layout: DwdpExpertLayout,
        num_moe_layers: int,
        param_shapes: Dict[str, torch.Size],
        param_dtypes: Dict[str, torch.dtype],
        device: torch.device,
        use_vmm: bool = False,
    ):
        self.layout = layout
        self.num_moe_layers = num_moe_layers
        self.param_shapes = param_shapes
        self.param_dtypes = param_dtypes
        self.device = device
        self.use_vmm = use_vmm

        self.prefetch_stream = torch.cuda.Stream(device=device)

        num_prefetch_experts = layout.num_prefetch_experts
        num_experts_per_worker = layout.num_experts_per_worker
        dwdp_size = layout.dwdp_size
        dwdp_rank = layout.dwdp_rank

        # Per-param metadata for standard (contiguous) weight layout
        # Expert dim is 0 for Triton kernels: [num_experts, ...]
        self.per_expert_bytes: Dict[str, int] = {}
        self.prefetch_view_shapes: Dict[str, torch.Size] = {}
        for name, shape in param_shapes.items():
            itemsize = param_dtypes[name].itemsize
            self.per_expert_bytes[name] = shape[1:].numel() * itemsize if len(shape) > 1 else itemsize
            view_shape = list(shape)
            view_shape[0] = num_prefetch_experts
            self.prefetch_view_shapes[name] = torch.Size(view_shape)

        # Allocate 2 buffer slots (ping-pong) for concat fallback mode.
        # In VMM mode, concat buffers are not needed — P2P copies go
        # directly into VMM slots.
        self.buffers: List[Dict[str, List[Optional[torch.Tensor]]]] = []
        if not use_vmm:
            for buf_idx in range(2):
                buffer = {}
                for param_name in param_shapes:
                    tensor_list: List[Optional[torch.Tensor]] = [None] * dwdp_size
                    for peer_rank in range(dwdp_size):
                        if peer_rank != dwdp_rank:
                            view_shape = self.prefetch_view_shapes[param_name]
                            tensor_list[peer_rank] = torch.empty(
                                view_shape,
                                dtype=param_dtypes[param_name],
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

        # Per-layer timing events for profiling (kept as per-layer arrays)
        num_slots_per_buffer = math.ceil(num_moe_layers / 2)
        self.prefetch_start_events: List[List[torch.cuda.Event]] = [
            [torch.cuda.Event(enable_timing=True) for _ in range(num_slots_per_buffer)]
            for _ in range(2)
        ]
        self.prefetch_end_events: List[List[torch.cuda.Event]] = [
            [torch.cuda.Event(enable_timing=True) for _ in range(num_slots_per_buffer)]
            for _ in range(2)
        ]
        self.compute_start_events: List[List[torch.cuda.Event]] = [
            [torch.cuda.Event(enable_timing=True) for _ in range(num_slots_per_buffer)]
            for _ in range(2)
        ]
        self.compute_end_events: List[List[torch.cuda.Event]] = [
            [torch.cuda.Event(enable_timing=True) for _ in range(num_slots_per_buffer)]
            for _ in range(2)
        ]

        # Per-layer CPU wall-clock timing for blocking sync
        self._sync_wall_ms: Dict[int, float] = {}
        self._ipc_open_ms: Dict[int, float] = {}
        self._ipc_close_ms: Dict[int, float] = {}

        # CudaRTLibrary instance for P2P operations
        from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
        self._cuda_lib = CudaRTLibrary()

        # Layer handle collectors and exchanged IPC handles, keyed by moe_idx.
        # Set by DwdpManager.init_prefetch_buffers() after IPC exchange.
        self._layer_collectors: Dict[int, DwdpLayerHandleCollector] = {}
        self._all_handles_by_layer: Dict[int, list] = {}

        # IPC handle cache for fast-path prefetch (no per-layer open/close).
        # Set by DwdpManager.init_prefetch_buffers() after IPC exchange.
        self._ipc_cache: Optional[DwdpIpcHandleCache] = None

        # VMM ping-pong slots (2 entries: slot 0 for even, slot 1 for odd).
        # When set, P2P copies write directly into VMM slot regions instead of
        # separate buffer tensors. Set by DwdpManager._init_vmm_buffers().
        self._vmm_slots: List[DwdpVmmWeightBuffer] = []

        logger.info(
            f"DwdpPrefetchBuffer allocated (P2P mode, vmm={use_vmm}): "
            f"num_prefetch_experts={num_prefetch_experts}, "
            f"num_slots_per_buffer={num_slots_per_buffer}, "
            f"per_expert_bytes={self.per_expert_bytes}"
        )

    def set_layer_collectors(
        self,
        collectors: Dict[int, DwdpLayerHandleCollector],
        all_handles_by_layer: Dict[int, list],
        ipc_cache: Optional[DwdpIpcHandleCache] = None,
    ) -> None:
        """Set reference to layer handle collectors, exchanged IPC handles, and IPC cache.

        Called by DwdpManager after IPC exchange completes.
        All dicts are keyed by moe_layer_idx (0-based).

        When ipc_cache is provided and all handles are cached, prefetch_layer()
        skips per-layer open/close and uses cached pointers directly.
        When ipc_cache is None or in sliding window mode, falls back to
        per-layer open/close via the collector.
        """
        self._layer_collectors = collectors
        self._all_handles_by_layer = all_handles_by_layer
        self._ipc_cache = ipc_cache

    def set_vmm_slots(self, slots: List[DwdpVmmWeightBuffer]) -> None:
        """Set VMM ping-pong slots for direct P2P-to-VMM-pool copy mode.

        When VMM slots are set, prefetch_layer() writes P2P copies directly
        into the VMM slot regions instead of separate buffer tensors.
        Local expert weights are also copied into the slot on the prefetch
        stream before P2P copies begin.
        """
        self._vmm_slots = slots

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
        local_weights: Dict[str, torch.Tensor],
    ) -> None:
        """Fetch peer expert weights via CUDA IPC P2P over NVLink.

        Event-driven pipeline (Fix 2): no CPU-side synchronize().
        P2P copies run on prefetch_stream and overlap with attention + MoE
        on the default stream. Stream ordering is enforced via per-slot
        CUDA events:
        - WAR: prefetch_stream waits on consume_events[buf_idx] before
          overwriting the slot (ensures prior compute is done).
        - RAW: prefetch_events[buf_idx] is recorded after P2P copy;
          default stream waits on it in wait_for_prefetch().

        When IPC handles are cached (DwdpIpcHandleCache with all handles open),
        this method uses cached mapped pointers directly (fast path).
        """
        buf_idx = moe_layer_idx % 2
        layer_slot = moe_layer_idx // 2
        layout = self.layout
        dwdp_rank = layout.dwdp_rank

        # Determine if we can use cached IPC handles
        use_cached = (
            self._ipc_cache is not None
            and self._ipc_cache.is_all_open
        )

        if not use_cached:
            # Slow path: per-layer IPC open (still needed for sliding window mode)
            collector = self._layer_collectors[moe_layer_idx]
            all_handles = self._all_handles_by_layer[moe_layer_idx]
            t_ipc_open_start = time.perf_counter_ns()
            collector.open_peer_handles(all_handles, dwdp_rank)
            t_ipc_open_end = time.perf_counter_ns()
            self._ipc_open_ms[moe_layer_idx] = (t_ipc_open_end - t_ipc_open_start) / 1e6

        # Select VMM slot (ping-pong)
        vmm_slot = self._vmm_slots[buf_idx] if self._vmm_slots else None

        # --- P2P copy on prefetch stream (event-driven, no synchronize) ---
        with torch.cuda.stream(self.prefetch_stream):
            # Record prefetch start on GPU (profiling)
            self.prefetch_start_events[buf_idx][layer_slot].record(self.prefetch_stream)

            # WAR: wait for previous use of this buffer slot to finish.
            # consume_events[buf_idx] fires after all prior default-stream work
            # that read from this slot has completed.
            self.prefetch_stream.wait_event(self.consume_events[buf_idx])

            # Copy local expert weights into VMM slot (overlaps with prev compute)
            if vmm_slot is not None:
                vmm_slot.copy_local_experts(local_weights)

            # Async P2P copies from peer GPU memory to local buffers
            stream_handle = self.prefetch_stream.cuda_stream

            for param_name in local_weights:
                per_expert_size = self.per_expert_bytes[param_name]
                num_prefetch = layout.num_prefetch_experts

                for peer_rank in range(layout.dwdp_size):
                    if peer_rank == dwdp_rank:
                        continue

                    # Calculate source pointer (peer's weight tensor + expert offset)
                    src_expert_offset = layout.get_prefetch_src_offset(peer_rank)
                    src_byte_offset = src_expert_offset * per_expert_size
                    copy_bytes = num_prefetch * per_expert_size

                    if use_cached:
                        # Fast path: use cached IPC pointer
                        src_ptr_int = self._ipc_cache.get_peer_ptr(
                            moe_layer_idx, peer_rank, param_name
                        )
                        src_ptr = ctypes.c_void_p(src_ptr_int + src_byte_offset)
                    else:
                        # Slow path: use collector's per-layer opened pointer
                        src_ptr = ctypes.c_void_p(
                            collector.peer_base_ptrs[(peer_rank, param_name)] + src_byte_offset
                        )

                    if vmm_slot is not None:
                        # VMM mode: copy directly into the composite VA slot region
                        dst_ptr_int, _ = vmm_slot.get_pool_ptr(
                            param_name, peer_rank, layout
                        )
                        dst_ptr = ctypes.c_void_p(dst_ptr_int)
                    else:
                        # Concat mode: copy into separate buffer tensor
                        dst_ptr = ctypes.c_void_p(
                            self.buffers[buf_idx][param_name][peer_rank].data_ptr()
                        )

                    assert src_ptr.value is not None and src_ptr.value != 0, (
                        f"Null src_ptr for peer={peer_rank} param={param_name}"
                    )
                    assert dst_ptr.value is not None and dst_ptr.value != 0, (
                        f"Null dst_ptr for peer={peer_rank} param={param_name}"
                    )

                    self._cuda_lib.cudaMemcpyAsync(
                        dst_ptr, src_ptr, copy_bytes, stream_handle
                    )

            # Record prefetch end on GPU (profiling)
            self.prefetch_end_events[buf_idx][layer_slot].record(self.prefetch_stream)

            # RAW: signal that prefetch for this slot is done.
            # Default stream will wait on this in wait_for_prefetch().
            self.prefetch_events[buf_idx].record(self.prefetch_stream)

        if use_cached:
            # Fast path: no sync needed, IPC handles stay open
            self._ipc_open_ms[moe_layer_idx] = 0.0
            self._sync_wall_ms[moe_layer_idx] = 0.0
            self._ipc_close_ms[moe_layer_idx] = 0.0
        else:
            # Slow path: must synchronize before closing IPC handles.
            # This breaks the overlap but is required for correctness
            # when handles are not cached.
            t_sync_start = time.perf_counter_ns()
            self.prefetch_stream.synchronize()
            t_sync_end = time.perf_counter_ns()
            self._sync_wall_ms[moe_layer_idx] = (t_sync_end - t_sync_start) / 1e6

            t_ipc_close_start = time.perf_counter_ns()
            collector.close_peer_handles()
            t_ipc_close_end = time.perf_counter_ns()
            self._ipc_close_ms[moe_layer_idx] = (t_ipc_close_end - t_ipc_close_start) / 1e6

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

    def record_compute_done(self, moe_layer_idx: int) -> None:
        """Record compute completion on the default stream (profiling only).

        In the event-driven pipeline, WAR protection is handled by
        consume_events recorded in wait_for_prefetch(). This method
        only records the compute_end timing event for profiling.
        """
        buf_idx = moe_layer_idx % 2
        layer_slot = moe_layer_idx // 2
        current_stream = torch.cuda.current_stream(self.device)
        self.compute_end_events[buf_idx][layer_slot].record(current_stream)

    def record_compute_start(self, moe_layer_idx: int) -> None:
        """Record compute start on the default stream."""
        buf_idx = moe_layer_idx % 2
        layer_slot = moe_layer_idx // 2
        current_stream = torch.cuda.current_stream(self.device)
        self.compute_start_events[buf_idx][layer_slot].record(current_stream)

    def get_buffer_views(
        self, moe_layer_idx: int
    ) -> Dict[str, List[Optional[torch.Tensor]]]:
        """Return buffer tensor views for the given MoE layer."""
        buf_idx = moe_layer_idx % 2
        return self.buffers[buf_idx]

    def get_layer_timing(self, moe_layer_idx: int) -> Dict[str, float]:
        """Return per-layer timing data for profiling.

        Returns dict with:
        - prefetch_gpu_ms: GPU time for P2P copy (CUDA event elapsed)
        - compute_gpu_ms: GPU time for MoE compute (CUDA event elapsed)
        - ipc_open_ms: CPU wall-clock for IPC handle open
        - sync_wall_ms: CPU wall-clock for prefetch_stream.synchronize()
        - ipc_close_ms: CPU wall-clock for IPC handle close
        - overlap_ratio: min(compute, prefetch_next) / prefetch_next
          (only meaningful when comparing layer N compute vs layer N+2 prefetch)
        """
        buf_idx = moe_layer_idx % 2
        layer_slot = moe_layer_idx // 2

        result: Dict[str, float] = {
            "prefetch_gpu_ms": 0.0,
            "compute_gpu_ms": 0.0,
            "ipc_open_ms": self._ipc_open_ms.get(moe_layer_idx, 0.0),
            "sync_wall_ms": self._sync_wall_ms.get(moe_layer_idx, 0.0),
            "ipc_close_ms": self._ipc_close_ms.get(moe_layer_idx, 0.0),
        }

        # GPU timing via CUDA events (requires synchronization)
        try:
            result["prefetch_gpu_ms"] = self.prefetch_start_events[buf_idx][
                layer_slot
            ].elapsed_time(self.prefetch_end_events[buf_idx][layer_slot])
        except Exception:
            pass

        try:
            result["compute_gpu_ms"] = self.compute_start_events[buf_idx][
                layer_slot
            ].elapsed_time(self.compute_end_events[buf_idx][layer_slot])
        except Exception:
            pass

        return result

    def dump_layer_timing(self, path: str) -> None:
        """Dump per-layer timing data to JSON file."""
        data = {}
        for moe_idx in range(self.num_moe_layers):
            data[f"layer_{moe_idx}"] = self.get_layer_timing(moe_idx)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Layer timing dumped to {path}")

    def cleanup(self) -> None:
        """Release prefetch buffers and events."""
        self.buffers.clear()
        self.prefetch_events.clear()
        self.consume_events.clear()
        self._layer_collectors.clear()
        self._all_handles_by_layer.clear()
        self._ipc_cache = None
        self._vmm_slots.clear()
        self.prefetch_stream = None
        self._cuda_lib = None
