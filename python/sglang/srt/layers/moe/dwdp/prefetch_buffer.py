"""Double-buffered async prefetch system for DWDP.

Uses torch.distributed all_gather (NCCL) to fetch peer expert weights via
NVLink, with a dedicated CUDA stream and ping-pong buffers to overlap
buffer copies with MoE compute.

Key design: all_gather runs on the default stream (NCCL requirement);
buffer copies run on the dedicated prefetch stream for overlap.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from sglang.srt.layers.moe.dwdp.dwdp_manager import DwdpExpertLayout

logger = logging.getLogger(__name__)


class DwdpPrefetchBuffer:
    """Double-buffered (ping-pong) async prefetch for DWDP expert weights.

    Buffer slot 0 is used by even-indexed MoE layers (0, 2, 4, ...),
    and slot 1 by odd-indexed (1, 3, 5, ...).

    Uses all_gather via torch.distributed to fetch peer weights over NVLink
    instead of CUDA IPC, which has compatibility issues on some hardware.
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

        logger.info(
            f"DwdpPrefetchBuffer allocated: "
            f"num_prefetch_experts={num_prefetch_experts}, "
            f"num_slots_per_buffer={num_slots_per_buffer}, "
            f"per_expert_bytes={self.per_expert_bytes}"
        )

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
        """Fetch peer expert weights via all_gather over NVLink.

        Uses torch.distributed all_gather to collect each rank's local
        expert weights, then copies the peer portion into the prefetch buffer.
        This avoids CUDA IPC which has compatibility issues on some hardware.
        """
        from sglang.srt.distributed.parallel_state import get_dwdp_group

        buf_idx = moe_layer_idx % 2
        layer_slot = moe_layer_idx // 2
        layout = self.layout
        dwdp_rank = layout.dwdp_rank
        group = get_dwdp_group()

        # --- Phase A: all_gather on the DEFAULT stream (NCCL) ---
        # NCCL collectives must run on the default stream; using a
        # non-default stream can cause illegal-address errors.
        # gloo (cpu_group) cannot handle CUDA tensors at all.
        gathered_per_param: Dict[str, List[torch.Tensor]] = {}
        for param_name, local_tensor in local_weights.items():
            # NCCL doesn't support float8 dtypes — view as uint8 (same
            # byte layout, itemsize == 1) for the collective.
            if local_tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                comm_tensor = local_tensor.contiguous().view(torch.uint8)
            else:
                comm_tensor = local_tensor.contiguous()

            gathered = [torch.empty_like(comm_tensor) for _ in range(layout.dwdp_size)]
            dist.all_gather(gathered, comm_tensor, group=group.device_group)
            gathered_per_param[param_name] = gathered

        # --- Phase B: copy peer data into prefetch buffers on the
        #     dedicated prefetch stream (overlaps with next compute) ---
        wait_compute_slot = layer_slot - 1 if moe_layer_idx >= 2 else None

        with torch.cuda.stream(self.prefetch_stream):
            if wait_compute_slot is not None and wait_compute_slot >= 0:
                self.prefetch_stream.wait_event(
                    self.compute_events[buf_idx][wait_compute_slot]
                )

            for param_name in local_weights:
                gathered = gathered_per_param[param_name]
                dst_dtype = self.param_dtypes[param_name]
                for peer_rank in range(layout.dwdp_size):
                    if peer_rank == dwdp_rank:
                        continue
                    src_expert_offset = layout.get_prefetch_src_offset(peer_rank)
                    num_prefetch = layout.num_prefetch_experts
                    dst_tensor = self.buffers[buf_idx][param_name][peer_rank]
                    peer_data = gathered[peer_rank]
                    # View back from uint8 to original dtype if needed
                    if peer_data.dtype == torch.uint8 and dst_dtype != torch.uint8:
                        peer_data = peer_data.view(dst_dtype)
                    dst_tensor.copy_(
                        peer_data[src_expert_offset:src_expert_offset + num_prefetch]
                    )

            self.prefetch_events[buf_idx][layer_slot].record(self.prefetch_stream)

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
        self.prefetch_stream = None
