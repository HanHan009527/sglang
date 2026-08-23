"""Focused tests for current UnifiedRadixCache batched HiCache PP sync."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    BatchedPPSyncCapability,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Holder:
    _uses_batched_hicache_pp_sync = UnifiedRadixCache._uses_batched_hicache_pp_sync
    _count_ready_acks = UnifiedRadixCache._count_ready_acks
    _sync_hicache_completion_counts = UnifiedRadixCache._sync_hicache_completion_counts
    _validated_hicache_acks = UnifiedRadixCache._validated_hicache_acks
    writing_check = UnifiedRadixCache.writing_check
    loading_check = UnifiedRadixCache.loading_check


class _Event:
    def __init__(self, ready=True):
        self.ready = ready
        self.synchronized = False

    def query(self):
        return self.ready

    def synchronize(self):
        self.synchronized = True


class _Ack:
    def __init__(self, node_ids, finish_event, num_tokens=7, timing_enabled=False):
        self.node_ids = node_ids
        self.finish_event = finish_event
        self.start_event = mock.Mock()
        self.num_tokens = num_tokens
        self.timing_enabled = timing_enabled


class TestUnifiedRadixCachePPSyncCurrent(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        values = dict(
            hicache_storage_backend=None,
            hicache_write_policy="write_through",
            enable_dp_attention=False,
            dp_size=1,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _holder(write_acks=(), load_acks=(), write_ids=(), load_ids=()):
        holder = _Holder()
        holder._hicache_pp_sync_mode = "batched"
        holder._hicache_pp_sync_counts = torch.zeros(2, dtype=torch.int32)
        holder.pp_rank = 0
        holder.pp_size = 2
        holder.work_list = []
        holder.cache_controller = SimpleNamespace(
            ack_write_queue=list(write_acks), ack_load_queue=list(load_acks)
        )
        holder.ongoing_write_through = {node_id: object() for node_id in write_ids}
        holder.ongoing_load_back = {
            node_id: (object(), object(), object()) for node_id in load_ids
        }
        holder.enable_storage = False
        holder.enable_storage_metrics = False
        holder.storage_metrics_collector = None
        holder._all_reduce = mock.Mock()
        holder._drain_async_work = mock.Mock()
        holder.dec_lock_ref = mock.Mock()
        holder.dec_host_lock_ref = mock.Mock()
        holder._finish_write_through_ack = mock.Mock(
            side_effect=lambda node_id: holder.ongoing_write_through.pop(node_id)
        )
        return holder

    def test_default_mode_is_legacy(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            envs.SGLANG_HICACHE_PP_SYNC_MODE.clear()
            self.assertEqual(envs.SGLANG_HICACHE_PP_SYNC_MODE.get(), "legacy")

    def test_batched_gate_is_narrow_and_does_not_create_group(self):
        holder = _Holder()
        holder.pp_size = 2
        holder._hicache_pp_sync_capability = BatchedPPSyncCapability.DEEPSEEK_V4
        params = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=mock.Mock(
                    return_value=object.__new__(DeepSeekV4TokenToKVPool)
                )
            )
        )
        with envs.SGLANG_HICACHE_PP_SYNC_MODE.override("batched"), mock.patch.object(
            torch.distributed, "new_group"
        ) as new_group:
            UnifiedRadixCache._init_hicache_pp_sync_mode(holder, self._args(), params)
        new_group.assert_not_called()
        self.assertEqual(holder._hicache_pp_sync_mode, "batched")
        self.assertEqual(holder._hicache_pp_sync_counts.dtype, torch.int32)
        self.assertEqual(holder._hicache_pp_sync_counts.device.type, "cpu")

    def test_batched_gate_accepts_regular_dsa_stack_capability(self):
        holder = _Holder()
        holder.pp_size = 2
        holder._hicache_pp_sync_capability = BatchedPPSyncCapability.DSA_KV_INDEXER
        params = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=mock.Mock(return_value=object.__new__(DSATokenToKVPool))
            )
        )

        with envs.SGLANG_HICACHE_PP_SYNC_MODE.override("batched"):
            UnifiedRadixCache._init_hicache_pp_sync_mode(holder, self._args(), params)

        self.assertEqual(holder._hicache_pp_sync_mode, "batched")
        self.assertIs(
            holder._hicache_pp_sync_capability,
            BatchedPPSyncCapability.DSA_KV_INDEXER,
        )

    def test_batched_gate_rejects_every_other_unsupported_dimension(self):
        unsupported = (
            ("bad", self._args()),
            ("unsupported HiCache stack for object", self._args()),
            ("PP size <= 1", self._args()),
            ("L3 enabled", self._args(hicache_storage_backend="eic")),
            ("write policy", self._args(hicache_write_policy="write_back")),
            ("DP attention enabled", self._args(enable_dp_attention=True)),
            ("DP size 2", self._args(dp_size=2)),
        )
        for reason, args in unsupported:
            params = SimpleNamespace(
                token_to_kv_pool_allocator=SimpleNamespace(
                    get_kvcache=mock.Mock(
                        return_value=object.__new__(DeepSeekV4TokenToKVPool)
                    )
                )
            )
            holder = _Holder()
            holder.pp_size = 2
            holder._hicache_pp_sync_capability = BatchedPPSyncCapability.DEEPSEEK_V4
            if reason == "PP size <= 1":
                holder.pp_size = 1
            if reason == "unsupported HiCache stack for object":
                params.token_to_kv_pool_allocator.get_kvcache.return_value = object()
                holder._hicache_pp_sync_capability = None
            mode = "bad" if reason == "bad" else "batched"
            expected = "legacy.*batched" if reason == "bad" else reason
            with self.subTest(reason=reason), envs.SGLANG_HICACHE_PP_SYNC_MODE.override(
                mode
            ), self.assertRaisesRegex(ValueError, expected):
                UnifiedRadixCache._init_hicache_pp_sync_mode(holder, args, params)

    def test_completion_uses_one_vector_collective_and_pp0_readiness(self):
        write_event = _Event(True)
        load_event = _Event(True)
        holder = self._holder(
            write_acks=[_Ack([11], write_event)],
            load_acks=[_Ack([12], load_event)],
            write_ids=[11],
            load_ids=[12],
        )

        UnifiedRadixCache._sync_hicache_completion_counts(holder)

        holder._all_reduce.assert_called_once()
        counts, op = holder._all_reduce.call_args.args
        torch.testing.assert_close(counts, torch.tensor([1, 1], dtype=torch.int32))
        self.assertEqual(op, torch.distributed.ReduceOp.MIN)
        self.assertEqual(
            holder._count_ready_acks(holder.cache_controller.ack_write_queue),
            1,
        )

    def test_empty_queues_still_participate_in_one_vector_collective(self):
        holder = self._holder()

        self.assertEqual(
            UnifiedRadixCache._sync_hicache_completion_counts(holder), (0, 0)
        )
        holder._all_reduce.assert_called_once()

    def test_nonzero_pp_rank_uses_propagated_counts_without_querying_events(self):
        write_event = _Event(False)
        load_event = _Event(False)
        holder = self._holder(
            # Node IDs are rank-local. The protocol propagates the PP0 prefix
            # counts, then each stage consumes the same-size local prefix.
            write_acks=[_Ack([21], write_event)],
            load_acks=[_Ack([22], load_event)],
            write_ids=[21],
            load_ids=[22],
        )
        holder.pp_rank = 1
        holder._all_reduce.side_effect = lambda counts, _op: counts.copy_(
            torch.tensor([1, 1], dtype=torch.int32)
        )

        self.assertEqual(
            UnifiedRadixCache._sync_hicache_completion_counts(holder), (1, 1)
        )
        self.assertFalse(write_event.synchronized)
        self.assertFalse(load_event.synchronized)

    def test_batched_check_waits_events_then_finishes_treecore_and_load_metrics(self):
        write_event = _Event(True)
        load_event = _Event(False)
        holder = self._holder(
            write_acks=[_Ack([11], write_event)],
            load_acks=[_Ack([12], load_event, num_tokens=13, timing_enabled=True)],
            write_ids=[11],
            load_ids=[12],
        )
        holder.metrics_collector = mock.Mock()
        holder.pp_rank = 1
        call_order = []
        write_event.synchronize = lambda: call_order.append("write_wait")
        load_event.synchronize = lambda: call_order.append("load_wait")
        holder._finish_write_through_ack.side_effect = (
            lambda node_id: call_order.append(("write_finish", node_id))
        )
        holder.dec_lock_ref.side_effect = lambda *_: call_order.append("device_unlock")
        holder.dec_host_lock_ref.side_effect = lambda *_: call_order.append(
            "host_unlock"
        )
        holder._all_reduce.side_effect = lambda counts, _op: counts.copy_(
            torch.tensor([1, 1], dtype=torch.int32)
        )
        holder.cache_controller.ack_load_queue[
            0
        ].start_event.elapsed_time.return_value = 250.0

        UnifiedRadixCache.check_hicache_events(holder)

        self.assertEqual(
            call_order,
            [
                "write_wait",
                ("write_finish", 11),
                "load_wait",
                "device_unlock",
                "host_unlock",
            ],
        )
        holder._finish_write_through_ack.assert_called_once_with(11)
        holder.metrics_collector.increment_load_back_num_tokens.assert_called_once_with(
            13
        )
        holder.metrics_collector.observe_load_back_duration.assert_called_once_with(
            0.25
        )
        self.assertEqual(holder.cache_controller.ack_write_queue, [])
        self.assertEqual(holder.cache_controller.ack_load_queue, [])

    def test_validation_fails_before_queue_mutation(self):
        event = _Event(True)
        holder = self._holder(write_acks=[_Ack([99], event)])
        with self.assertRaisesRegex(RuntimeError, "ACK IDs diverged"):
            UnifiedRadixCache._validated_hicache_acks(
                holder, holder.cache_controller.ack_write_queue, {}, 1, kind="write"
            )
        self.assertEqual(len(holder.cache_controller.ack_write_queue), 1)
        self.assertFalse(event.synchronized)

    def test_duplicate_ack_ids_fail_before_mutation(self):
        event = _Event(True)
        holder = self._holder(write_acks=[_Ack([41, 41], event)], write_ids=[41])

        with self.assertRaisesRegex(RuntimeError, "ACK IDs repeated"):
            UnifiedRadixCache._validated_hicache_acks(
                holder,
                holder.cache_controller.ack_write_queue,
                holder.ongoing_write_through,
                1,
                kind="write",
            )

        self.assertEqual(len(holder.cache_controller.ack_write_queue), 1)
        self.assertIn(41, holder.ongoing_write_through)
        self.assertFalse(event.synchronized)

    def test_short_queue_fails_before_mutation(self):
        holder = self._holder(write_ids=[41])

        with self.assertRaisesRegex(RuntimeError, "write ACK queue diverged"):
            UnifiedRadixCache._validated_hicache_acks(
                holder,
                holder.cache_controller.ack_write_queue,
                holder.ongoing_write_through,
                1,
                kind="write",
            )

        self.assertEqual(holder.cache_controller.ack_write_queue, [])
        self.assertIn(41, holder.ongoing_write_through)

    def test_legacy_mode_keeps_existing_write_then_load_path(self):
        holder = self._holder()
        holder._hicache_pp_sync_mode = "legacy"
        holder.writing_check = mock.Mock()
        holder.loading_check = mock.Mock()

        UnifiedRadixCache.check_hicache_events(holder)

        holder._drain_async_work.assert_called_once()
        holder.writing_check.assert_called_once_with()
        holder.loading_check.assert_called_once_with()
        holder._all_reduce.assert_not_called()


if __name__ == "__main__":
    unittest.main()
