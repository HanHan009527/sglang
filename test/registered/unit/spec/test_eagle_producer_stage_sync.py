"""Tests for the default-off EAGLE producer-stage sync diagnostic."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.environ import envs
from sglang.srt.speculative.eagle_worker_v2 import (
    _debug_eagle_producer_stage_sync,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_ROOT = Path(__file__).parents[4]


def _make_worker_and_batch():
    worker = SimpleNamespace(
        device="cuda",
        gpu_id=7,
        ps=SimpleNamespace(pp_rank=1, dp_rank=2, attn_dp_rank=2, tp_rank=3),
    )
    batch = SimpleNamespace(
        batch_size=Mock(return_value=4),
        forward_mode=SimpleNamespace(name="DECODE"),
    )
    return worker, batch


class TestEagleProducerStageSync(unittest.TestCase):
    def test_flag_defaults_off_and_does_not_resolve_stream(self):
        worker, batch = _make_worker_and_batch()
        with patch(
            "sglang.srt.speculative.eagle_worker_v2.torch.get_device_module"
        ) as get_device_module:
            _debug_eagle_producer_stage_sync(
                worker, stage="target_verify_complete", batch=batch, cuda_graph=True
            )
        get_device_module.assert_not_called()
        self.assertFalse(hasattr(worker, "_eagle_producer_stage_sync_state"))

    def test_enabled_sync_records_completed_host_event(self):
        worker, batch = _make_worker_and_batch()
        stream = Mock()
        device_module = SimpleNamespace(current_stream=Mock(return_value=stream))
        with (
            envs.SGLANG_DEBUG_EAGLE_PRODUCER_STAGE_SYNC.override(True),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.torch.get_device_module",
                return_value=device_module,
            ),
        ):
            _debug_eagle_producer_stage_sync(
                worker, stage="draft_extend_complete", batch=batch, cuda_graph=True
            )

        stream.synchronize.assert_called_once_with()
        event = worker._eagle_producer_stage_sync_state["ring"][-1]
        self.assertEqual(event["stage"], "draft_extend_complete")
        self.assertEqual(event["status"], "complete")
        self.assertEqual(event["pp_rank"], 1)
        self.assertEqual(event["attn_dp_rank"], 2)
        self.assertEqual(event["batch_size"], 4)
        self.assertIs(event["cuda_graph"], True)
        self.assertIsInstance(event["elapsed_us"], int)

    def test_sync_failure_records_stage_and_reraises_without_tensor_access(self):
        worker, batch = _make_worker_and_batch()
        stream = Mock()
        stream.synchronize.side_effect = RuntimeError("asynchronous launch failure")
        device_module = SimpleNamespace(current_stream=Mock(return_value=stream))
        with (
            envs.SGLANG_DEBUG_EAGLE_PRODUCER_STAGE_SYNC.override(True),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.torch.get_device_module",
                return_value=device_module,
            ),
            self.assertRaisesRegex(RuntimeError, "asynchronous launch failure"),
        ):
            _debug_eagle_producer_stage_sync(
                worker, stage="tail_draft_complete", batch=batch, cuda_graph=False
            )

        event = worker._eagle_producer_stage_sync_state["ring"][-1]
        self.assertEqual(event["stage"], "tail_draft_complete")
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["error_type"], "RuntimeError")

    def test_call_sites_are_after_stage_calls(self):
        source = (
            _ROOT / "python/sglang/srt/speculative/eagle_worker_v2.py"
        ).read_text()
        verify_call = source.index("batch_output = self.verify(")
        verify_sync = source.index('stage="target_verify_complete"', verify_call)
        extend_call = source.index(
            "self.draft_worker._draft_extend_for_decode(batch, batch_output)"
        )
        extend_sync = source.index('stage="draft_extend_complete"', extend_call)
        tail_call = source.index("self.draft_worker.draft(batch)", extend_sync)
        tail_sync = source.index('stage="tail_draft_complete"', tail_call)

        self.assertLess(verify_call, verify_sync)
        self.assertLess(verify_sync, extend_call)
        self.assertLess(extend_call, extend_sync)
        self.assertLess(extend_sync, tail_call)
        self.assertLess(tail_call, tail_sync)


if __name__ == "__main__":
    unittest.main()
