from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboDPAttentionPreparer
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.metrics_collector import DPCooperationInfo
from sglang.srt.utils.common import require_mlp_tp_gather

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.managers.scheduler import Scheduler


_ENABLE_METRICS_DP_ATTENTION = envs.SGLANG_ENABLE_METRICS_DP_ATTENTION.get()


@dataclass
class MLPSyncBatchInfo:
    dp_size: int
    tp_size: int
    cp_size: int

    num_tokens: int
    num_tokens_for_logprob: int
    can_cuda_graph: bool
    is_extend_in_batch: bool
    local_can_run_tbo: bool
    local_forward_mode: int

    # some gathered elements
    tp0_info: torch.Tensor = None
    global_num_tokens: list[int] = None
    global_num_tokens_for_logprob: list[int] = None
    tbo_split_seq_index: torch.Tensor = None
    global_forward_mode: int = None
    dp_cooperation_info: Optional[DPCooperationInfo] = None

    def _get_local_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        return torch.tensor(
            [
                self.num_tokens,
                self.num_tokens_for_logprob,
                int(self.can_cuda_graph),
                int(self.is_extend_in_batch),
                int(self.local_can_run_tbo),
                self.local_forward_mode,
            ],
            device=device,
            dtype=dtype,
        )

    def _get_fallback_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        return torch.tensor(
            [
                0,  # num_tokens
                0,  # num_tokens_for_logprob
                1,  # can_cuda_graph
                0,  # is_extend_in_batch
                1,  # local_can_run_tbo
                ForwardMode.IDLE.value,  # local_forward_mode
            ],
            device=device,
            dtype=dtype,
        )

    # For async AllGather (Phase 2 fuse_dp_gather)
    _async_handle: Optional[object] = None  # torch.distributed.Work handle
    _async_global_info_tensor: Optional[torch.Tensor] = None
    _async_device: Optional[str] = None

    def _process_gathered_info(self, global_info_tensor, device):
        """Extract per-DP-rank metadata from the gathered info tensor."""
        if device == "cpu":
            tp_active_ranks = get_tp_group().active_ranks_cpu
        else:
            tp_active_ranks = get_tp_group().active_ranks

        # Set fallback values for inactive ranks
        tp_info = global_info_tensor.view(self.dp_size * self.tp_size * self.cp_size, 6)
        tp_info[tp_active_ranks == 0] = self._get_fallback_tensor(device=device)

        tp0_info = global_info_tensor[:, 0, :]
        self.tp0_info = tp0_info
        # Perform only one Device-to-Host (D2H) memory copy
        cpu_data = tp0_info[:, :2].cpu()
        self.global_num_tokens = cpu_data[:, 0].tolist()
        self.global_num_tokens_for_logprob = cpu_data[:, 1].tolist()
        self.can_cuda_graph = bool(tp0_info[:, 2].min().item())
        self.is_extend_in_batch = bool(tp0_info[:, 3].max().item())
        if _ENABLE_METRICS_DP_ATTENTION:
            self.dp_cooperation_info = DPCooperationInfo.create(tp0_info[:, 5].tolist())

    def all_gather(self, device, group: torch.distributed.ProcessGroup):
        local_info_tensor = self._get_local_tensor(device=device)
        global_info_tensor = torch.empty(
            (self.dp_size, self.tp_size * self.cp_size, 6),
            dtype=torch.int64,
            device=device,
        )

        torch.distributed.all_gather_into_tensor(
            global_info_tensor.flatten(),
            local_info_tensor,
            group=group,
        )
        self._process_gathered_info(global_info_tensor, device)

    def all_gather_async(self, device, group: torch.distributed.ProcessGroup):
        """Launch an async AllGather and return immediately with conservative defaults.

        The caller must later call _all_gather_finalize() to wait for completion
        and extract the exact per-DP-rank metadata.
        """
        local_info_tensor = self._get_local_tensor(device=device)
        self._async_global_info_tensor = torch.empty(
            (self.dp_size, self.tp_size * self.cp_size, 6),
            dtype=torch.int64,
            device=device,
        )
        self._async_handle = torch.distributed.all_gather_into_tensor(
            self._async_global_info_tensor.flatten(),
            local_info_tensor,
            group=group,
            async_op=True,
        )
        self._async_device = device

        # Set conservative defaults so the scheduler can proceed immediately.
        # Each rank assumes all DP ranks have the same token count as local.
        self.global_num_tokens = [self.num_tokens] * self.dp_size
        self.global_num_tokens_for_logprob = [self.num_tokens_for_logprob] * self.dp_size
        self.can_cuda_graph = True  # conservative: allow cuda graph
        self.is_extend_in_batch = self.is_extend_in_batch
        # tp0_info not available yet; set to None
        self.tp0_info = None

    def _all_gather_finalize(self):
        """Wait for the async AllGather to complete and extract exact values.

        Returns the exact global_num_tokens list.
        """
        if self._async_handle is None:
            return self.global_num_tokens
        self._async_handle.wait()
        self._process_gathered_info(self._async_global_info_tensor, self._async_device)
        self._async_handle = None
        self._async_global_info_tensor = None
        self._async_device = None
        return self.global_num_tokens


def _update_gather_batch(
    batch: ScheduleBatch,
    mlp_sync_info: MLPSyncBatchInfo,
    require_mlp_tp_gather: bool,
    skip_all_gather=False,
):
    # TODO: handle the case when moe_dense_tp_size != 1
    if not require_mlp_tp_gather:
        batch.global_num_tokens = [mlp_sync_info.num_tokens]
        batch.global_num_tokens_for_logprob = [mlp_sync_info.num_tokens_for_logprob]
    else:
        batch.global_num_tokens = mlp_sync_info.global_num_tokens
        batch.global_num_tokens_for_logprob = (
            mlp_sync_info.global_num_tokens_for_logprob
        )
    if not skip_all_gather:
        batch.is_extend_in_batch = mlp_sync_info.is_extend_in_batch
        batch.tbo_split_seq_index = mlp_sync_info.tbo_split_seq_index
        batch.global_forward_mode = mlp_sync_info.global_forward_mode

    # Check forward mode for cuda graph
    batch.can_run_dp_cuda_graph = mlp_sync_info.can_cuda_graph

    # Propagate the async AllGather info so it can be finalized later
    if mlp_sync_info._async_handle is not None:
        batch._async_allgather_info = mlp_sync_info


def prepare_mlp_sync_batch_raw(
    local_batch: ScheduleBatch,
    dp_size: int,
    attn_tp_size: int,
    attn_cp_size: int,
    tp_group: GroupCoordinator,
    get_idle_batch: Callable[[], ScheduleBatch],
    disable_cuda_graph: bool,
    require_mlp_tp_gather: bool,
    disable_overlap_schedule: bool,
    offload_tags: set[str],
):
    # Check if other DP workers have running batches
    if local_batch is None or local_batch.forward_mode.is_prebuilt():
        num_tokens = 0
        num_tokens_for_logprob = 0
    elif local_batch.forward_mode.is_decode():
        num_tokens = local_batch.batch_size()
        num_tokens_for_logprob = num_tokens
    else:
        num_tokens = local_batch.extend_num_tokens
        num_tokens_for_logprob = sum(
            # We should have at least 1 token for sample in every case.
            max(extend_len - logprob_start_len, 1)
            for logprob_start_len, extend_len in zip(
                local_batch.extend_logprob_start_lens,
                local_batch.extend_lens,
            )
        )
        assert (
            local_batch.return_logprob
            or num_tokens_for_logprob == local_batch.batch_size()
        )

    skip_all_gather = envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get()
    can_cuda_graph = (
        local_batch is None
        or local_batch.forward_mode.is_decode_or_idle()
        or local_batch.forward_mode.is_prebuilt()
    ) and not disable_cuda_graph

    is_extend_in_batch = local_batch.forward_mode.is_extend() if local_batch else False
    if local_batch is not None:
        local_batch.is_extend_in_batch = is_extend_in_batch

    tbo_preparer = TboDPAttentionPreparer()
    if len(offload_tags) == 0 and (
        disable_overlap_schedule
        or envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get()
    ):
        group = tp_group.device_group
        device = tp_group.device
    else:
        group = tp_group.cpu_group
        device = "cpu"

    local_can_run_tbo, local_forward_mode = tbo_preparer.prepare_all_gather(local_batch)

    mlp_sync_info = MLPSyncBatchInfo(
        dp_size=dp_size,
        tp_size=attn_tp_size,
        cp_size=attn_cp_size,
        num_tokens=num_tokens,
        num_tokens_for_logprob=num_tokens_for_logprob,
        can_cuda_graph=can_cuda_graph,
        is_extend_in_batch=is_extend_in_batch,
        local_can_run_tbo=local_can_run_tbo,
        local_forward_mode=local_forward_mode,
    )

    fuse_dp_gather = (
        envs.SGLANG_DEEPEP_FUSE_DP_GATHER.get()
        and not get_moe_a2a_backend().is_none()
    )

    if not skip_all_gather and not fuse_dp_gather:
        mlp_sync_info.all_gather(device=device, group=group)

        mlp_sync_info.tbo_split_seq_index, mlp_sync_info.global_forward_mode = (
            tbo_preparer.compute_output(
                mlp_sync_info.tp0_info[:, 4:6],
            )
        )
    elif fuse_dp_gather:
        # Phase 2: Launch async AllGather — don't block the scheduler.
        # Conservative defaults are set immediately so scheduling decisions
        # can proceed.  The exact per-DP-rank metadata will be extracted
        # later in ForwardBatch.init_new() before the model runner needs it.
        mlp_sync_info.all_gather_async(device=device, group=group)
        # Skip TBO in fused mode (requires global consensus we don't have yet)
        mlp_sync_info.tbo_split_seq_index = None
        mlp_sync_info.global_forward_mode = local_forward_mode

    need_idle_batch = skip_all_gather or max(mlp_sync_info.global_num_tokens) > 0
    # With async AllGather, conservative defaults may underestimate other ranks'
    # token counts.  Always create an idle batch when fuse_dp_gather is active
    # and the local rank has no tokens, because other DP ranks may need us
    # to participate in EP dispatch.
    if fuse_dp_gather and not need_idle_batch and local_batch is None:
        need_idle_batch = True
    if need_idle_batch:
        batch_to_gather = local_batch
        if local_batch is None:
            batch_to_gather = local_batch = get_idle_batch()
        elif local_batch.forward_mode.is_prebuilt():
            # NOTE: for prebuilt batch, we add an inner idle batch to run MLP sync
            batch_to_gather = local_batch.inner_idle_batch = get_idle_batch()
        _update_gather_batch(
            batch_to_gather,
            mlp_sync_info,
            require_mlp_tp_gather,
            skip_all_gather and not fuse_dp_gather,
        )

    if _ENABLE_METRICS_DP_ATTENTION and local_batch is not None:
        local_batch.dp_cooperation_info = mlp_sync_info.dp_cooperation_info

    return local_batch


class SchedulerDPAttnMixin:
    def prepare_mlp_sync_batch(self: Scheduler, local_batch: ScheduleBatch):
        return prepare_mlp_sync_batch_raw(
            local_batch,
            dp_size=self.server_args.dp_size,
            attn_tp_size=self.attn_tp_size,
            attn_cp_size=self.attn_cp_size,
            tp_group=self.tp_group,
            get_idle_batch=self.get_idle_batch,
            disable_cuda_graph=self.server_args.disable_cuda_graph,
            require_mlp_tp_gather=require_mlp_tp_gather(self.server_args),
            disable_overlap_schedule=self.server_args.disable_overlap_schedule,
            offload_tags=self.offload_tags,
        )

    def maybe_prepare_mlp_sync_batch(
        self: Scheduler,
        batch: Optional[ScheduleBatch],
        need_sync: Optional[bool] = None,
    ) -> Optional[ScheduleBatch]:
        """
        Helper to prepare MLP sync batch for DP attention.
        Should be called after get_new_batch_prefill().

        Args:
            batch: The batch to process
            need_sync: If specified, overrides self.require_mlp_sync for prepare_mlp_sync_batch decision
        """
        if need_sync if need_sync is not None else self.require_mlp_sync:
            batch = self.prepare_mlp_sync_batch(batch)
        return batch

    def get_idle_batch(self: Scheduler) -> ScheduleBatch:
        idle_batch = ScheduleBatch.init_new(
            [],
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        idle_batch.prepare_for_idle()
        return idle_batch
