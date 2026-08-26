import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.dsa.dsa_indexer import (
    _topk_transform_with_diagnostic_capacity,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    _probe_topk_v2_inputs,
    _probe_topk_v2_raw_output,
    _topk_transform_v2_paged,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSATopKV2DiagnosticContract(unittest.TestCase):
    def _metadata(self):
        return SimpleNamespace(
            real_page_table=torch.tensor(
                [
                    [99, 99, 99],
                    [1, 99, 99],
                    [2, 3, 99],
                ],
                dtype=torch.int32,
            ),
            topk_v2_plan=torch.zeros((4, 2), dtype=torch.int32),
            page_size=4,
        )

    @staticmethod
    def _record_asserts(events):
        def record(condition, message):
            events.append((bool(condition), message))

        return record

    def test_default_path_passes_no_raw_output_and_queues_no_probe(self):
        metadata = self._metadata()
        logits = torch.zeros((3, 8), dtype=torch.float32)
        lengths = torch.tensor([0, 2, 5], dtype=torch.int32)
        transformed = torch.full((3, 4), -1, dtype=torch.int32)

        with (
            patch(
                "sglang.srt.layers.attention.dsa.dsa_topk_backend.envs."
                "SGLANG_ENABLE_ASYNC_ASSERT.get",
                return_value=False,
            ),
            patch.object(torch, "_assert_async") as async_assert,
            patch(
                "sglang.kernels.ops.attention.dsv4.topk." "topk_transform_512_v2",
                side_effect=lambda *args: args[3].copy_(transformed),
            ) as topk_v2,
        ):
            actual = _topk_transform_v2_paged(
                logits,
                lengths,
                4,
                metadata,
            )

        self.assertTrue(torch.equal(actual, transformed))
        self.assertEqual(len(topk_v2.call_args.args), 6)
        async_assert.assert_not_called()

    def test_indexer_passes_active_buffer_capacity_only_when_enabled(self):
        logits = torch.empty((2, 8), dtype=torch.float32)
        for enabled, expected_kwargs in (
            (False, {}),
            (True, {"diagnostic_page_capacity": 17}),
        ):
            metadata = SimpleNamespace(
                topk_transform=Mock(return_value=torch.empty((2, 4)))
            )
            with patch(
                "sglang.srt.layers.attention.dsa.dsa_indexer.envs."
                "SGLANG_ENABLE_ASYNC_ASSERT.get",
                return_value=enabled,
            ):
                _topk_transform_with_diagnostic_capacity(
                    metadata=metadata, logits=logits, topk=4, page_capacity=17
                )
            with self.subTest(enabled=enabled):
                metadata.topk_transform.assert_called_once_with(
                    logits, 4, **expected_kwargs
                )

    def test_enabled_wrapper_uses_same_launch_raw_output(self):
        metadata = self._metadata()
        logits = torch.zeros((3, 8), dtype=torch.float32)
        lengths = torch.tensor([0, 2, 5], dtype=torch.int32)
        raw = torch.tensor(
            [[-1, -1, -1, -1], [0, 1, -1, -1], [0, 1, 2, 4]],
            dtype=torch.int32,
        )
        transformed = torch.tensor(
            [[-1, -1, -1, -1], [4, 5, -1, -1], [8, 9, 10, 12]],
            dtype=torch.int32,
        )
        events = []

        def run_topk(*args):
            events.append(("topk",))
            args[3].copy_(transformed)
            args[6].copy_(raw)

        with (
            patch(
                "sglang.srt.layers.attention.dsa.dsa_topk_backend.envs."
                "SGLANG_ENABLE_ASYNC_ASSERT.get",
                return_value=True,
            ),
            patch.object(
                torch, "_assert_async", side_effect=self._record_asserts(events)
            ) as async_assert,
            patch(
                "sglang.kernels.ops.attention.dsv4.topk." "topk_transform_512_v2",
                side_effect=run_topk,
            ) as topk_v2,
        ):
            actual = _topk_transform_v2_paged(
                logits,
                lengths,
                4,
                metadata,
                diagnostic_page_capacity=4,
            )

        self.assertTrue(torch.equal(actual, transformed))
        self.assertIsNotNone(topk_v2.call_args.args[6])
        self.assertEqual(events[5], ("topk",))
        self.assertEqual(async_assert.call_count, 9)
        self.assertTrue(all(event[0] for event in events[:5]))
        self.assertTrue(all(event[0] for event in events[6:]))

    def test_input_probe_ignores_stale_tail_but_classifies_live_pages(self):
        lengths = torch.tensor([0, 2, 5], dtype=torch.int32)
        base_table = self._metadata().real_page_table

        for name, page_table, expected in (
            ("stale_tail_is_ignored", base_table, [True] * 5),
            ("zero_live_page", base_table.clone(), [True] * 5),
            ("negative_live_page", base_table.clone(), [True, True, True, False, True]),
            ("page_at_capacity", base_table.clone(), [True, True, True, True, False]),
        ):
            if name == "zero_live_page":
                page_table[1, 0] = 0
            elif name == "negative_live_page":
                page_table[1, 0] = -1
            elif name == "page_at_capacity":
                page_table[2, 1] = 4
            events = []
            with patch.object(
                torch, "_assert_async", side_effect=self._record_asserts(events)
            ):
                _probe_topk_v2_inputs(
                    lengths=lengths,
                    score_width=8,
                    page_table=page_table,
                    page_size=4,
                    page_capacity=4,
                )
            with self.subTest(name=name):
                self.assertEqual([event[0] for event in events], expected)

    def test_input_probe_classifies_length_contracts(self):
        cases = {
            "negative_length": (
                torch.tensor([-1], dtype=torch.int32),
                torch.ones((1, 2), dtype=torch.int32),
                [False, True, True, True, True],
            ),
            "past_score_width": (
                torch.tensor([9], dtype=torch.int32),
                torch.ones((1, 3), dtype=torch.int32),
                [True, False, True, True, True],
            ),
            "past_page_table_width": (
                torch.tensor([8], dtype=torch.int32),
                torch.ones((1, 1), dtype=torch.int32),
                [True, True, False, True, True],
            ),
        }

        for name, (lengths, page_table, expected) in cases.items():
            events = []
            with patch.object(
                torch, "_assert_async", side_effect=self._record_asserts(events)
            ):
                _probe_topk_v2_inputs(
                    lengths=lengths,
                    score_width=8,
                    page_table=page_table,
                    page_size=4,
                    page_capacity=4,
                )
            with self.subTest(name=name):
                self.assertEqual([event[0] for event in events], expected)

    def test_raw_probe_distinguishes_selected_prefix_and_padding(self):
        lengths = torch.tensor([0, 2, 5], dtype=torch.int32)
        transformed = torch.tensor(
            [[-1, -1, -1, -1], [4, 5, -1, -1], [8, 9, 10, 12]],
            dtype=torch.int32,
        )
        valid = torch.tensor(
            [[-1, -1, -1, -1], [0, 1, -1, -1], [0, 1, 2, 4]],
            dtype=torch.int32,
        )

        cases = {
            "valid": (valid, [True, True, True, True]),
            "negative_selected": (valid.clone(), [False, True, True, False]),
            "past_live_prefix": (valid.clone(), [True, False, True, True]),
            "non_sentinel_padding": (valid.clone(), [True, True, False, False]),
        }
        cases["negative_selected"][0][1, 0] = -1
        cases["past_live_prefix"][0][1, 1] = 2
        cases["non_sentinel_padding"][0][1, 2] = 0

        for name, (raw, expected) in cases.items():
            events = []
            with patch.object(
                torch, "_assert_async", side_effect=self._record_asserts(events)
            ):
                _probe_topk_v2_raw_output(
                    raw_indices=raw,
                    transformed_indices=transformed,
                    lengths=lengths,
                )
            with self.subTest(name=name):
                self.assertEqual([event[0] for event in events], expected)


if __name__ == "__main__":
    unittest.main()
