from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.async_kv_utils import (
    AsyncInfo,
    AsyncTransferItem,
    StreamAsyncSubmitter,
    TransferKVChunkSet,
    cached_group_concurrent_contiguous,
    env_int,
)
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, kv_to_page_indices

logger = logging.getLogger(__name__)


class MooncakeKVAsyncMixin:
    """Mixin that implements layerwise async KV transfer for MooncakeKVManager.

    This mixin is intentionally self-contained and only relies on a small set
    of attributes/methods provided by the concrete manager:

    Required attrs:
    - `disaggregation_mode`, `is_mla_backend`, `attn_tp_size`
    - `kv_args` (with kv/state ptrs and state_type)
    - `transfer_infos`, `request_status`, `decode_kv_args_table`

    Required methods:
    - `_transfer_data(session_id, blocks)`
    - `_execute_async_transfer_item(...)`
    """

    # -------------------------
    # Lifecycle / state
    # -------------------------

    def _async_kv_init_state(self) -> None:
        self._async_kv_enabled: bool = False
        self._async_submitter: Optional[StreamAsyncSubmitter] = None
        self._notify_queue: Optional[deque[AsyncInfo]] = None
        self._waiting_rooms: Optional[deque[Optional[TransferKVChunkSet]]] = None
        self._current_kv_chunk_infos: Optional[TransferKVChunkSet] = None
        self._req_begin_count: Dict[int, deque[int]] = {}
        self._req_bids: Dict[int, deque[int]] = {}
        self._req_tensor_seen: Dict[int, set[AsyncTransferItem]] = {}
        self._room_to_kv_chunk_info: Dict[int, tuple[TransferKVChunkSet, int]] = {}
        self._lock: Optional[threading.Lock] = None
        self._bids_cond: Optional[threading.Condition] = None
        self._queue_lock: Optional[threading.Lock] = None
        self._kv_tensor_ntensors: int = 0
        self._state_tensor_ntensors: int = 0
        self._tensor_ntensors_total: int = 0
        self._kv_cache_nlayers: int = 0
        self._async_kv_missing_wait_ms: int = 0
        self._layer_ready_events: Dict[Tuple[int, AsyncTransferItem], Any] = {}

    @property
    def async_kv_enabled(self) -> bool:
        return bool(self._async_kv_enabled)

    def _async_kv_enable(self) -> None:
        """Enable layerwise async KV transfers for PREFILL."""

        # Only supported in PREFILL.
        if getattr(self, "disaggregation_mode", None) is None:
            return
        from sglang.srt.disaggregation.utils import DisaggregationMode

        if self.disaggregation_mode != DisaggregationMode.PREFILL:
            return

        self._async_kv_enabled = True
        self._notify_queue = deque()
        self._waiting_rooms = deque()
        self._current_kv_chunk_infos = None
        self._req_begin_count = {}
        self._req_bids = {}
        self._req_tensor_seen = {}
        self._room_to_kv_chunk_info = {}
        # _lock protects per-room bookkeeping (_req_*, _room_to_kv_chunk_info, _layer_ready_events).
        self._lock = threading.Lock()
        self._bids_cond = threading.Condition(self._lock)
        self._queue_lock = threading.Lock()
        self._kv_tensor_ntensors = len(self.kv_args.kv_data_ptrs)
        self._state_tensor_ntensors = len(self.kv_args.state_data_ptrs)
        self._tensor_ntensors_total = self._kv_tensor_ntensors + self._state_tensor_ntensors
        self._kv_cache_nlayers = (
            self._kv_tensor_ntensors
            if self.is_mla_backend
            else (self._kv_tensor_ntensors // 2)
        )
        self._async_kv_missing_wait_ms = env_int(
            "SGLANG_ASYNC_KV_MISSING_WAIT_MS", "20"
        )
        self._layer_ready_events = {}

        self._async_submitter = StreamAsyncSubmitter(self._async_put_kvcache_func)

    # -------------------------
    # Scheduler hook
    # -------------------------

    def _async_prepare_batch_if_eligible(self, sch: Any, batch: Any) -> bool:
        if not self._async_kv_enabled:
            return False

        eligible_reqs = [
            req
            for req in batch.reqs
            if getattr(req, "bootstrap_host", None) != FAKE_BOOTSTRAP_HOST
        ]
        if not eligible_reqs:
            return False

        # Current async path only supports non-chunked, full-send (start_send_idx=0) requests.
        if not all(
            getattr(req, "start_send_idx", None) == 0
            and getattr(req, "is_chunked", 0) <= 0
            for req in eligible_reqs
        ):
            return False

        self._async_prepare_batch(sch, batch)
        return True

    def maybe_prepare_async_kv_split(self, sch: Any, batch: Any) -> Optional[Any]:
        """Prepare a split-prefill async driver for a scheduler batch.

        The driver reuses the existing async transfer machinery, but lets the
        split-prefill outer loop explicitly notify completed layer ranges instead
        of relying on the attention backend to emit per-layer callbacks.
        """

        if not self._async_prepare_batch_if_eligible(sch, batch):
            return None
        return self

    # -------------------------
    # Internal helpers
    # -------------------------

    def _async_put_kvcache_func(self) -> None:
        if self._notify_queue is None or self._queue_lock is None:
            return
        with self._queue_lock:
            if not self._notify_queue:
                return
            info = self._notify_queue.pop()
        self._async_put_kv_cache_internal(info)

    def _async_try_sync_ready_event(
        self, *, room_id: int, transfer_item: AsyncTransferItem
    ) -> None:
        if not self._async_kv_enabled or self._lock is None:
            return
        event_key = (int(room_id), transfer_item)
        with self._lock:
            event = self._layer_ready_events.pop(event_key, None)
        if event is None:
            return

        if not torch.cuda.is_available():
            return
        event.synchronize()

    def _async_try_record_ready_event_for_rooms(
        self, *, rooms: Tuple[int, ...], transfer_item: AsyncTransferItem
    ) -> None:
        if not self._async_kv_enabled or self._lock is None:
            return
        if not torch.cuda.is_available():
            return
        event = torch.cuda.Event(enable_timing=False, blocking=False, interprocess=False)
        event.record()
        with self._lock:
            for rid in rooms:
                self._layer_ready_events[(int(rid), transfer_item)] = event

    def _async_maybe_start_next_kv_chunk(self) -> None:
        if not self._async_kv_enabled or self._queue_lock is None or self._lock is None:
            return
        assert self._async_submitter is not None
        begin_count = self._async_submitter.get_step_count()
        with self._queue_lock:
            assert self._waiting_rooms is not None
            current = self._waiting_rooms.pop() if self._waiting_rooms else None
            self._current_kv_chunk_infos = current

        if not current:
            logger.warning("async kv layer0: no waiting rooms")
            return

        # Keep lock ordering consistent: _queue_lock -> _lock.
        with self._lock:
            for idx, rid in enumerate(current.rooms):
                if rid not in self._req_begin_count:
                    self._req_begin_count[rid] = deque()
                self._req_begin_count[rid].appendleft(begin_count)
                self._room_to_kv_chunk_info[rid] = (current, idx)

    def _async_filter_current_kv_chunk_infos(self) -> None:
        if not self._async_kv_enabled or self._queue_lock is None or self._lock is None:
            return
        with self._queue_lock:
            current = self._current_kv_chunk_infos
            if not current or not current.rooms:
                return
            rooms = current.rooms

        keep_indices = [
            idx
            for idx, rid in enumerate(rooms)
            if rid in self.transfer_infos and self.request_status.get(rid) != KVPoll.Success
        ]
        if not keep_indices or len(keep_indices) == len(rooms):
            return

        filtered = TransferKVChunkSet(
            rooms=tuple(rooms[i] for i in keep_indices),
            prefill_kv_indices=tuple(current.prefill_kv_indices[i] for i in keep_indices),
            index_slices=tuple(current.index_slices[i] for i in keep_indices),
            prefill_state_indices=tuple(current.prefill_state_indices[i] for i in keep_indices),
        )
        with self._queue_lock:
            self._current_kv_chunk_infos = filtered

        with self._lock:
            for rid in rooms:
                if rid not in filtered.rooms:
                    self._room_to_kv_chunk_info.pop(rid, None)
            for idx, rid in enumerate(filtered.rooms):
                self._room_to_kv_chunk_info[rid] = (filtered, idx)

    def _async_get_info_with_risk(self, room: int) -> dict:
        if room not in self.transfer_infos:
            status = self.request_status.get(room)
            if status != KVPoll.Success:
                logger.warning(
                    "async kv skip: room=%s not in transfer_infos status=%s",
                    room,
                    status,
                )
            return {}
        return self.transfer_infos[room]

    def _async_submit_layer(
        self,
        session_id: str,
        src_ptr: int,
        dst_ptr: int,
        prefill_kv_blocks: npt.NDArray[np.int64],
        dst_kv_blocks: npt.NDArray[np.int64],
        item_len: int,
    ) -> int:
        prefill_kv_blocks_tmp, dst_kv_blocks_tmp = cached_group_concurrent_contiguous(
            prefill_kv_blocks, dst_kv_blocks
        )
        if not prefill_kv_blocks_tmp:
            return 0
        transfer_blocks = []
        for prefill_index, decode_index in zip(prefill_kv_blocks_tmp, dst_kv_blocks_tmp):
            src_addr = src_ptr + int(prefill_index[0]) * item_len
            dst_addr = dst_ptr + int(decode_index[0]) * item_len
            transfer_blocks.append((src_addr, dst_addr, item_len * len(prefill_index)))
        return int(self._transfer_data(session_id, transfer_blocks))

    def _async_put_kv_cache_internal(self, async_info: AsyncInfo) -> None:
        kv_chunk_info = async_info.kv_chunk_info
        if not kv_chunk_info.rooms:
            return
        infos = [self._async_get_info_with_risk(room) for room in kv_chunk_info.rooms]
        for transfer_item in async_info.transfer_items:
            for room_id, transfer_info_dict, kv_indice, index_slice, prefill_state_idx in zip(
                kv_chunk_info.rooms,
                infos,
                kv_chunk_info.prefill_kv_indices,
                kv_chunk_info.index_slices,
                kv_chunk_info.prefill_state_indices,
            ):
                if not transfer_info_dict:
                    continue
                for transfer_info in transfer_info_dict.values():
                    if transfer_info.is_dummy:
                        continue
                    session_id = transfer_info.mooncake_session_id
                    registration = self.decode_kv_args_table.get(session_id)
                    if registration is None:
                        logger.warning(
                            "async kv skip: missing registration room=%s session=%s item=%s:%s",
                            room_id,
                            session_id,
                            transfer_item.pool,
                            transfer_item.tensor_idx,
                        )
                        continue

                    self._async_try_sync_ready_event(
                        room_id=int(room_id), transfer_item=transfer_item
                    )
                    status = self._execute_async_transfer_item(
                        transfer_item=transfer_item,
                        transfer_info=transfer_info,
                        registration=registration,
                        prefill_kv_indices=kv_indice,
                        index_slice=index_slice,
                        prefill_state_idx=int(prefill_state_idx),
                    )

                    assert self._bids_cond is not None
                    with self._bids_cond:
                        self._req_tensor_seen.setdefault(room_id, set()).add(transfer_item)
                        self._req_bids.setdefault(room_id, deque()).appendleft(int(status))
                        self._bids_cond.notify_all()

    def _async_mark_transfer_item_ready(self, transfer_item: AsyncTransferItem) -> None:
        if not self._async_kv_enabled:
            return

        assert self._queue_lock is not None
        with self._queue_lock:
            current = self._current_kv_chunk_infos

        if current is None:
            self._async_maybe_start_next_kv_chunk()
            with self._queue_lock:
                current = self._current_kv_chunk_infos

        if transfer_item.pool == "kv":
            is_valid = 0 <= transfer_item.tensor_idx < self._kv_tensor_ntensors
        elif transfer_item.pool == "state":
            is_valid = 0 <= transfer_item.tensor_idx < self._state_tensor_ntensors
        else:
            is_valid = False

        if not is_valid:
            logger.warning(
                "async kv layer ready skipped: item=%s:%s",
                transfer_item.pool,
                transfer_item.tensor_idx,
            )
            return

        if current:
            self._async_filter_current_kv_chunk_infos()
            with self._queue_lock:
                current = self._current_kv_chunk_infos

        if not current or not current.rooms:
            return

        if transfer_item.pool == "kv":
            self._async_try_record_ready_event_for_rooms(
                rooms=current.rooms, transfer_item=transfer_item
            )
        elif transfer_item.pool == "state" and self.kv_args.state_type == "mamba":
            self._async_try_record_ready_event_for_rooms(
                rooms=current.rooms, transfer_item=transfer_item
            )

        with self._queue_lock:
            assert self._notify_queue is not None
            self._notify_queue.appendleft(
                AsyncInfo(transfer_items=(transfer_item,), kv_chunk_info=current)
            )
        assert self._async_submitter is not None
        self._async_submitter.step_async()

    def _async_collect_split_ready_transfer_items(
        self,
        forward_batch: Any,
        start_layer: int,
        end_layer: int,
    ) -> Tuple[AsyncTransferItem, ...]:
        attn_backend = getattr(forward_batch, "attn_backend", None)
        token_to_kv_pool = getattr(forward_batch, "token_to_kv_pool", None)
        req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)

        if attn_backend is None or token_to_kv_pool is None or req_to_token_pool is None:
            logger.warning(
                "async kv split notify skipped: missing forward batch metadata for range [%s, %s)",
                start_layer,
                end_layer,
            )
            return ()

        full_attn_layers = set(getattr(attn_backend, "full_attn_layers", ()))
        full_layer_nums = getattr(token_to_kv_pool, "full_layer_nums", len(full_attn_layers))
        use_mla = bool(getattr(token_to_kv_pool, "use_mla", False))
        kv_ntensors = full_layer_nums if use_mla else full_layer_nums * 2
        full_layer_mapping = getattr(
            token_to_kv_pool, "full_attention_layer_id_mapping", {}
        )
        mamba_map = getattr(req_to_token_pool, "mamba_map", {})
        mamba_state_tensors_per_layer = int(
            getattr(attn_backend, "_mamba_state_tensors_per_layer", 0)
        )
        mamba_num_layers = int(
            getattr(attn_backend, "_mamba_num_layers", len(mamba_map) if mamba_map else 0)
        )

        transfer_items = []
        for layer_id in range(start_layer, end_layer):
            if layer_id in full_attn_layers:
                packed_id = full_layer_mapping.get(layer_id)
                if packed_id is None:
                    logger.warning(
                        "async kv split notify missing full-attn mapping: model_layer=%s",
                        layer_id,
                    )
                    continue
                transfer_items.append(
                    AsyncTransferItem(
                        pool="kv", tensor_idx=int(packed_id), model_layer_id=int(layer_id)
                    )
                )
                if not use_mla:
                    transfer_items.append(
                        AsyncTransferItem(
                            pool="kv",
                            tensor_idx=int(packed_id + full_layer_nums),
                            model_layer_id=int(layer_id),
                        )
                    )
                continue

            mamba_layer_idx = mamba_map.get(layer_id)
            if mamba_layer_idx is None or mamba_state_tensors_per_layer <= 0:
                logger.warning(
                    "async kv split notify missing linear-state mapping: model_layer=%s",
                    layer_id,
                )
                continue

            for tensor_idx in range(mamba_state_tensors_per_layer):
                transfer_items.append(
                    AsyncTransferItem(
                        pool="state",
                        tensor_idx=int(tensor_idx * mamba_num_layers + mamba_layer_idx),
                        model_layer_id=int(layer_id),
                    )
                )

        return tuple(transfer_items)

    def notify_split_range_ready(
        self,
        forward_batch: Any,
        start_layer: int,
        end_layer: int,
    ) -> None:
        """Notify async KV driver that a split-prefill layer range has completed."""

        if not self._async_kv_enabled or end_layer <= start_layer:
            return

        if not getattr(forward_batch, "async_kv_batch_started", False):
            forward_batch.async_kv_batch_started = True
            self._async_maybe_start_next_kv_chunk()

        transfer_items = self._async_collect_split_ready_transfer_items(
            forward_batch, start_layer, end_layer
        )
        for transfer_item in transfer_items:
            self._async_mark_transfer_item_ready(transfer_item)

    def _async_wait_for_bids(self, rid: int, *, timeout_s: Optional[float] = None) -> bool:
        if self._bids_cond is None:
            return False
        deadline = None if timeout_s is None else (time.time() + float(timeout_s))
        with self._bids_cond:
            while True:
                q = self._req_bids.get(rid)
                if q is not None and len(q) >= self._tensor_ntensors_total:
                    return True
                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return False
                    self._bids_cond.wait(timeout=remaining)
                else:
                    self._bids_cond.wait()

    def _async_resend_missing_state_tensors(
        self, room: int, missing_state_items: list[AsyncTransferItem]
    ) -> None:
        if not missing_state_items:
            return
        info = self._room_to_kv_chunk_info.get(room)
        if info is None:
            return
        kv_chunk_info, idx = info
        if idx >= len(kv_chunk_info.prefill_state_indices):
            return
        prefill_state_idx = kv_chunk_info.prefill_state_indices[idx]
        if prefill_state_idx is None or prefill_state_idx < 0:
            return
        transfer_info_dict = self.transfer_infos.get(room)
        if not transfer_info_dict:
            return
        for transfer_info in transfer_info_dict.values():
            if transfer_info.is_dummy:
                continue
            registration = self.decode_kv_args_table.get(transfer_info.mooncake_session_id)
            if registration is None or not transfer_info.dst_state_indices:
                continue
            for transfer_item in missing_state_items:
                self._execute_async_transfer_item(
                    transfer_item=transfer_item,
                    transfer_info=transfer_info,
                    registration=registration,
                    prefill_kv_indices=np.empty(0, dtype=np.int64),
                    index_slice=slice(0, 0),
                    prefill_state_idx=int(prefill_state_idx),
                )

    def _async_pop_req_bids(self, rid: int, is_remove: bool):
        assert self._bids_cond is not None
        with self._bids_cond:
            q = self._req_bids.pop(rid) if is_remove else self._req_bids[rid]
            return [q.pop() for _ in range(self._tensor_ntensors_total)]

    def _async_flush_all_layers(self, rid: int) -> None:
        if self._lock is None:
            return
        with self._lock:
            if rid not in self._req_begin_count:
                return

        assert self._async_submitter is not None
        while True:
            with self._lock:
                if not self._req_begin_count.get(rid):
                    break
                begin_count = self._req_begin_count[rid].pop()

            self._async_submitter.wait_sent_finish(begin_count + self._tensor_ntensors_total)
            self._async_wait_for_bids(rid)

            with self._lock:
                current_last = len(self._req_begin_count.get(rid, ())) == 0

            statuses = self._async_pop_req_bids(rid, current_last)
            if current_last:
                with self._lock:
                    seen = set(self._req_tensor_seen.get(rid, set()))
                missing_kv = [
                    AsyncTransferItem(pool="kv", tensor_idx=i)
                    for i in range(self._kv_tensor_ntensors)
                    if AsyncTransferItem(pool="kv", tensor_idx=i) not in seen
                ]
                missing_state = [
                    AsyncTransferItem(pool="state", tensor_idx=i)
                    for i in range(self._state_tensor_ntensors)
                    if AsyncTransferItem(pool="state", tensor_idx=i) not in seen
                ]
                if missing_state:
                    self._async_resend_missing_state_tensors(rid, missing_state)
                if any(s != 0 for s in statuses) or missing_kv or missing_state:
                    logger.warning(
                        "async kv flush: room=%s nonzero=%s missing=%s",
                        rid,
                        sum(1 for s in statuses if s != 0),
                        len(missing_kv) + len(missing_state),
                    )
                with self._lock:
                    self._req_tensor_seen.pop(rid, None)
                    self._room_to_kv_chunk_info.pop(rid, None)
                    if self._layer_ready_events:
                        keys = [k for k in self._layer_ready_events.keys() if k[0] == rid]
                        for k in keys:
                            self._layer_ready_events.pop(k, None)
                    self._req_begin_count.pop(rid, None)
                break

        with self._lock:
            self._req_begin_count.pop(rid, None)

    def _async_prepare_batch(self, sch: Any, batch: Any) -> None:
        if not self._async_kv_enabled:
            return

        rooms = []
        prefill_kv_indices = []
        index_slices = []
        prefill_state_indices = []

        for req in batch.reqs:
            if getattr(req, "is_chunked", 0) > 0:
                continue
            if getattr(req, "bootstrap_host", None) == FAKE_BOOTSTRAP_HOST:
                continue
            room = int(req.bootstrap_room)
            if not self._async_is_eligible_room(room):
                continue
            page_size = sch.token_to_kv_pool_allocator.page_size
            start_idx = req.start_send_idx
            end_idx = min(len(req.fill_ids), len(req.origin_input_ids))
            if end_idx <= start_idx:
                continue
            kv_indices = (
                sch.req_to_token_pool.req_to_token[req.req_pool_idx, start_idx:end_idx]
                .cpu()
                .numpy()
            )
            page_indices = kv_to_page_indices(kv_indices, page_size)
            if len(page_indices) == 0:
                continue
            index_slice = slice(
                req.disagg_kv_sender.curr_idx,
                req.disagg_kv_sender.curr_idx + len(page_indices),
            )
            rooms.append(room)
            prefill_kv_indices.append(page_indices)
            index_slices.append(index_slice)
            state_idx = -1
            if self.kv_args.state_type == "mamba":
                mapping = getattr(
                    sch.req_to_token_pool, "req_index_to_mamba_index_mapping", None
                )
                if mapping is not None:
                    mapped_state_idx = mapping[req.req_pool_idx]
                    state_idx = int(
                        mapped_state_idx.item()
                        if hasattr(mapped_state_idx, "item")
                        else mapped_state_idx
                    )
            prefill_state_indices.append(state_idx)

        kv_chunk_info_set = (
            TransferKVChunkSet(
                rooms=tuple(rooms),
                prefill_kv_indices=tuple(prefill_kv_indices),
                index_slices=tuple(index_slices),
                prefill_state_indices=tuple(prefill_state_indices),
            )
            if rooms
            else None
        )

        assert self._queue_lock is not None
        with self._queue_lock:
            assert self._waiting_rooms is not None
            self._waiting_rooms.appendleft(kv_chunk_info_set)

    def _async_is_eligible_room(self, room: int) -> bool:
        transfer_info_dict = self.transfer_infos.get(room)
        if transfer_info_dict is None or not transfer_info_dict:
            return False
        for transfer_info in transfer_info_dict.values():
            if transfer_info.is_dummy:
                continue
            registration = self.decode_kv_args_table.get(transfer_info.mooncake_session_id)
            if registration is None:
                return False
            if registration.dst_attn_tp_size != self.attn_tp_size:
                return False
        return not all(t.is_dummy for t in transfer_info_dict.values())

    def _async_use_for_room(self, room: int) -> bool:
        if not self._async_kv_enabled or self._lock is None:
            return False
        with self._lock:
            return room in self._req_begin_count
