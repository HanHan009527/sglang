import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.dsa_indexer import (
    Indexer,
    _make_eager_idle_topk_result,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_INDEXER = "sglang.srt.layers.attention.dsa.dsa_indexer"


class TestDSAIndexerIdle(unittest.TestCase):
    def test_builds_invalid_topk_rows_for_padded_idle_tokens(self):
        result = _make_eager_idle_topk_result(
            torch.empty((2, 16)),
            index_topk=8,
            return_indices=True,
        )

        self.assertEqual(result.shape, (2, 8))
        self.assertEqual(result.dtype, torch.int32)
        self.assertTrue(torch.all(result == -1))

    def test_returns_none_when_indices_are_not_consumed(self):
        self.assertIsNone(
            _make_eager_idle_topk_result(
                torch.empty((2, 16)),
                index_topk=8,
                return_indices=False,
            )
        )

    def _run_eager_idle(
        self,
        *,
        return_indices=True,
        forward_mode=ForwardMode.IDLE,
        original_forward_mode=None,
        symmetric_spec_moe_dummy=False,
    ):
        indexer = SimpleNamespace(index_topk=8)
        batch = SimpleNamespace(
            forward_mode=forward_mode,
            _original_forward_mode=original_forward_mode,
            symmetric_spec_moe_dummy=symmetric_spec_moe_dummy,
        )
        broadcast_calls = []
        capture_calls = []

        with (
            patch(f"{_INDEXER}._is_cuda", True),
            patch(f"{_INDEXER}.get_is_capture_mode", return_value=False),
            patch(
                f"{_INDEXER}._broadcast_indexer_topk_from_rank0",
                side_effect=lambda result: broadcast_calls.append(result) or result,
            ),
            patch(
                f"{_INDEXER}.maybe_capture_indexer_topk",
                side_effect=lambda layer_id, result: capture_calls.append(
                    (layer_id, result)
                )
                or result,
            ),
            patch(
                f"{_INDEXER}.get_attn_backend",
                side_effect=AssertionError("eager idle must not fetch metadata"),
            ),
        ):
            result = Indexer.forward_cuda(
                indexer,
                x=(torch.empty((2, 16)), torch.empty((2, 1))),
                q_lora=torch.empty((2, 16)),
                positions=torch.empty((2,), dtype=torch.int64),
                forward_batch=batch,
                layer_id=3,
                return_indices=return_indices,
            )
        self.assertEqual(len(broadcast_calls), 1)
        self.assertIs(broadcast_calls[0], result)
        self.assertEqual(len(capture_calls), 1)
        self.assertEqual(capture_calls[0][0], 3)
        self.assertIs(capture_calls[0][1], result)
        return result

    def test_eager_idle_short_circuits_before_paged_mqa(self):
        result = self._run_eager_idle()

        self.assertEqual(result.shape, (2, 8))
        self.assertTrue(torch.all(result == -1))

    def test_eager_idle_without_index_consumer_returns_none(self):
        self.assertIsNone(self._run_eager_idle(return_indices=False))

    def test_rewritten_target_verify_idle_short_circuits_without_draft_flag(self):
        result = self._run_eager_idle(
            forward_mode=ForwardMode.TARGET_VERIFY,
            original_forward_mode=ForwardMode.IDLE,
            # Target-verify padding is materialized before the model forward,
            # but unlike draft padding it is not tagged as an attention-bypass
            # row.  The original mode is the authoritative idle marker here.
            symmetric_spec_moe_dummy=False,
        )
        self.assertTrue(torch.all(result == -1))

    def test_rewritten_decode_idle_short_circuits_without_draft_flag(self):
        result = self._run_eager_idle(
            forward_mode=ForwardMode.DECODE,
            original_forward_mode=ForwardMode.IDLE,
            symmetric_spec_moe_dummy=False,
        )
        self.assertTrue(torch.all(result == -1))

    def _assert_existing_metadata_path(
        self,
        *,
        forward_mode,
        capture_mode,
        original_forward_mode=None,
        symmetric_spec_moe_dummy=False,
    ):
        indexer = SimpleNamespace(index_topk=8)
        batch = SimpleNamespace(
            forward_mode=forward_mode,
            _original_forward_mode=original_forward_mode,
            symmetric_spec_moe_dummy=symmetric_spec_moe_dummy,
        )

        with (
            patch(f"{_INDEXER}._is_cuda", True),
            patch(f"{_INDEXER}.get_is_capture_mode", return_value=capture_mode),
            patch(
                f"{_INDEXER}._is_in_piecewise_or_breakable_cuda_graph",
                return_value=False,
            ),
            patch(
                f"{_INDEXER}.get_attn_backend",
                side_effect=AssertionError("existing metadata path reached"),
            ),
            self.assertRaisesRegex(AssertionError, "existing metadata path reached"),
        ):
            Indexer.forward_cuda(
                indexer,
                x=torch.empty((2, 16)),
                q_lora=torch.empty((2, 16)),
                positions=torch.empty((2,), dtype=torch.int64),
                forward_batch=batch,
                layer_id=3,
            )

    def test_capture_idle_keeps_existing_metadata_path(self):
        self._assert_existing_metadata_path(
            forward_mode=ForwardMode.IDLE, capture_mode=True
        )

    def test_active_decode_keeps_existing_metadata_path(self):
        self._assert_existing_metadata_path(
            forward_mode=ForwardMode.DECODE, capture_mode=False
        )

    def test_active_target_verify_is_not_logical_idle(self):
        self._assert_existing_metadata_path(
            forward_mode=ForwardMode.TARGET_VERIFY,
            capture_mode=False,
            original_forward_mode=None,
            symmetric_spec_moe_dummy=False,
        )

    def test_rewritten_extend_is_not_logical_idle(self):
        self._assert_existing_metadata_path(
            forward_mode=ForwardMode.EXTEND,
            capture_mode=False,
            original_forward_mode=ForwardMode.IDLE,
            symmetric_spec_moe_dummy=True,
        )

    def test_rewritten_target_verify_capture_keeps_existing_metadata_path(self):
        self._assert_existing_metadata_path(
            forward_mode=ForwardMode.TARGET_VERIFY,
            capture_mode=True,
            original_forward_mode=ForwardMode.IDLE,
            symmetric_spec_moe_dummy=False,
        )

    def test_non_cuda_idle_keeps_existing_metadata_path(self):
        indexer = SimpleNamespace(index_topk=8)
        batch = SimpleNamespace(forward_mode=ForwardMode.IDLE)
        with (
            patch(f"{_INDEXER}._is_cuda", False),
            patch(f"{_INDEXER}.get_is_capture_mode", return_value=False),
            patch(
                f"{_INDEXER}._is_in_piecewise_or_breakable_cuda_graph",
                return_value=False,
            ),
            patch(
                f"{_INDEXER}.get_attn_backend",
                side_effect=AssertionError("existing metadata path reached"),
            ),
            self.assertRaisesRegex(AssertionError, "existing metadata path reached"),
        ):
            Indexer.forward_cuda(
                indexer,
                x=torch.empty((2, 16)),
                q_lora=torch.empty((2, 16)),
                positions=torch.empty((2,), dtype=torch.int64),
                forward_batch=batch,
                layer_id=3,
            )


if __name__ == "__main__":
    unittest.main()
