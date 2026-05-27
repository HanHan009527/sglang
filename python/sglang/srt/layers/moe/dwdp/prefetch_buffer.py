"""Double-buffered async prefetch system for DWDP.

Uses CUDA IPC P2P reads (cudaMemcpyAsync over NVLink) to fetch peer expert
weights, with a dedicated CUDA stream and ping-pong buffers to overlap
buffer copies with MoE compute.

Key design: unlike the previous NCCL all_gather approach, P2P copies run
entirely on the prefetch stream with no default-stream dependency and no
collective synchronization barrier. This enables independent rank execution
compatible with DP attention.
"""

from __future__ import annotations

import ctypes
import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.dwdp.dwdp_manager import DwdpExpertLayout, DwdpLayerHandleCollector

logger = logging.getLogger(__name__)


class DwdpPrefetchBuffer:
    """Double-buffered (ping-pong) async prefetch for DWDP expert weights.

    Buffer slot 0 is used by even-indexed MoE layers (0, 2, 4, ...),
    and slot 1 by odd-indexed (1, 3, 5, ...).

    Uses CUDA IPC P2P reads (cudaMemcpyAsync) to fetch peer expert weights
    over NVLink instead of NCCL all_gather. Each rank independently reads
    peer GPU memory — no collective synchronization is required.
    """

    def __init__(
        self,
        layout: DwdpExpertLayout,
        num_moe_layers: int,
        param_shapes: Dict[str, torch.Size],
        param_dtypes: Dict[str, torch.dtype],
        device: torch.device,
    ):
        self.layout = layout
        self.num_moe_layers = num_moe_layers
        self.param_shapes = param_shapes
        self.param_dtypes = param_dtypes
        self.device = device

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

        # Allocate 2 buffer slots (ping-pong)
        # buffers[buf_idx][param_name] = list of tensors, one per rank
        # Entry at dwdp_rank is None (local weights used directly)
        self.buffers: List[Dict[str, List[Optional[torch.Tensor]]]] = []
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

        # Per-layer CUDA events
        num_slots_per_buffer = math.ceil(num_moe_layers / 2)
        self.prefetch_events: List[List[torch.cuda.Event]] = [
            [torch.cuda.Event() for _ in range(num_slots_per_buffer)]
            for _ in range(2)
        ]
        self.compute_events: List[List[torch.cuda.Event]] = [
            [torch.cuda.Event() for _ in range(num_slots_per_buffer)]
            for _ in range(2)
        ]

        # CudaRTLibrary instance for P2P operations
        from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
        self._cuda_lib = CudaRTLibrary()

        # Layer handle collectors and exchanged IPC handles, keyed by moe_idx.
        # Set by DwdpManager.init_prefetch_buffers() after IPC exchange.
        self._layer_collectors: Dict[int, DwdpLayerHandleCollector] = {}
        self._all_handles_by_layer: Dict[int, list] = {}

        logger.info(
            f"DwdpPrefetchBuffer allocated (P2P mode): "
            f"num_prefetch_experts={num_prefetch_experts}, "
            f"num_slots_per_buffer={num_slots_per_buffer}, "
            f"per_expert_bytes={self.per_expert_bytes}"
        )

    def set_layer_collectors(
        self,
        collectors: Dict[int, DwdpLayerHandleCollector],
        all_handles_by_layer: Dict[int, list],
    ) -> None:
        """Set reference to layer handle collectors and exchanged IPC handles.

        Called by DwdpManager after IPC exchange completes.
        Both dicts are keyed by moe_layer_idx (0-based).
        """
        self._layer_collectors = collectors
        self._all_handles_by_layer = all_handles_by_layer

    def initialize_compute_events(self) -> None:
        """Pre-record first compute events so prefetch_first_layers() can proceed."""
        current_stream = torch.cuda.current_stream(self.device)
        for buf_idx in range(2):
            self.compute_events[buf_idx][0].record(current_stream)

    def prefetch_layer(
        self,
        moe_layer_idx: int,
        local_weights: Dict[str, torch.Tensor],
    ) -> None:
        """Fetch peer expert weights via CUDA IPC P2P over NVLink.

        Unlike the previous NCCL all_gather approach, this method:
        - Runs entirely on the prefetch stream (no default stream dependency)
        - Does NOT require peer ranks to participate (true P2P read)
        - Opens IPC handles per-call, closes after copy is enqueued
        - Transfers only the needed expert slice (not full local weights)
        - No FP8 dtype workaround needed (raw byte copy)
        """
        buf_idx = moe_layer_idx % 2
        layer_slot = moe_layer_idx // 2
        layout = self.layout
        dwdp_rank = layout.dwdp_rank

        # Get the handle collector for this layer
        collector = self._layer_collectors[moe_layer_idx]
        all_handles = self._all_handles_by_layer[moe_layer_idx]

        # Open IPC handles for peer weights (~4 params * (dwdp_size-1) mappings,
        # well under the CUDA driver IPC mapping limit of ~256)
        collector.open_peer_handles(all_handles, dwdp_rank)

        # --- P2P copy on prefetch stream ---
        wait_compute_slot = layer_slot - 1 if moe_layer_idx >= 2 else None

        with torch.cuda.stream(self.prefetch_stream):
            # Wait for previous layer's compute to finish (overlap window)
            if wait_compute_slot is not None and wait_compute_slot >= 0:
                self.prefetch_stream.wait_event(
                    self.compute_events[buf_idx][wait_compute_slot]
                )

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

                    src_ptr = ctypes.c_void_p(
                        collector.peer_base_ptrs[(peer_rank, param_name)] + src_byte_offset
                    )
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

            self.prefetch_events[buf_idx][layer_slot].record(self.prefetch_stream)

        # Close IPC handles immediately — cudaMemcpyAsync has already recorded
        # the source address in the CUDA command buffer, so the mapping is no
        # longer needed. This keeps concurrent IPC mappings minimal (~28).
        collector.close_peer_handles()

    def wait_for_prefetch(self, moe_layer_idx: int) -> None:
        """Default stream waits for prefetch of this layer to complete."""
        buf_idx = moe_layer_idx % 2
        layer_slot = moe_layer_idx // 2
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_event(self.prefetch_events[buf_idx][layer_slot])

    def record_compute_done(self, moe_layer_idx: int) -> None:
        """Record compute completion on the default stream."""
        buf_idx = moe_layer_idx % 2
        layer_slot = moe_layer_idx // 2
        current_stream = torch.cuda.current_stream(self.device)
        self.compute_events[buf_idx][layer_slot].record(current_stream)

    def get_buffer_views(
        self, moe_layer_idx: int
    ) -> Dict[str, List[Optional[torch.Tensor]]]:
        """Return buffer tensor views for the given MoE layer."""
        buf_idx = moe_layer_idx % 2
        return self.buffers[buf_idx]

    def cleanup(self) -> None:
        """Release prefetch buffers and events."""
        self.buffers.clear()
        self.prefetch_events.clear()
        self.compute_events.clear()
        self._layer_collectors.clear()
        self._all_handles_by_layer.clear()
        self.prefetch_stream = None
        self._cuda_lib = None
