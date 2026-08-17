import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EaglePPVerifyInputRaw,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_utils import TreeMaskMode
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPPEaglePrebuiltMerge(unittest.TestCase):
    @staticmethod
    def _draft_input(bonus_tokens):
        batch_size = bonus_tokens.shape[0]
        return EagleDraftInput(
            topk_p=torch.ones((batch_size, 1), dtype=torch.float32),
            topk_index=bonus_tokens.reshape(-1, 1),
            hidden_states=torch.zeros((batch_size, 4), dtype=torch.float32),
            bonus_tokens=bonus_tokens,
        )

    def test_new_pd_request_is_normalized_before_running_batch_merge(self):
        bonus_tokens = torch.tensor([101, 202], dtype=torch.int64)
        draft_input = self._draft_input(bonus_tokens)
        batch = SimpleNamespace(
            spec_algorithm=SimpleNamespace(
                build_disagg_draft_input=lambda *_args: draft_input
            ),
            reqs=[],
            device=torch.device("cpu"),
            enable_overlap=False,
            input_ids=torch.empty((0,), dtype=torch.int64),
            spec_info=None,
        )
        server_args = SimpleNamespace(
            pp_size=2,
            speculative_num_draft_tokens=4,
        )

        ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(
            batch,
            server_args,
            future_map=None,
        )

        self.assertTrue(torch.equal(batch.input_ids, bonus_tokens))
        self.assertIsInstance(batch.spec_info, EaglePPVerifyInputRaw)
        self.assertTrue(
            torch.equal(
                batch.spec_info.draft_tokens,
                torch.tensor([[101, 101, 101, 101], [202, 202, 202, 202]]),
            )
        )

        running_raw = EaglePPVerifyInputRaw(
            draft_tokens=torch.tensor([[11, 12, 13, 14]]),
            bonus_tokens=torch.tensor([11]),
            top_scores_index=torch.tensor([[0, 1, 2]]),
            parent_list=torch.tensor([[-1, 0, 1]]),
            accept_lens=torch.tensor([2]),
        )
        running_raw.merge_batch(batch.spec_info)
        self.assertEqual(running_raw.draft_tokens.shape[0], 3)
        self.assertTrue(
            torch.equal(running_raw.bonus_tokens, torch.tensor([11, 101, 202]))
        )

    def test_non_pp_keeps_eagle_draft_input(self):
        bonus_tokens = torch.tensor([303], dtype=torch.int64)
        draft_input = self._draft_input(bonus_tokens)
        batch = SimpleNamespace(
            spec_algorithm=SimpleNamespace(
                build_disagg_draft_input=lambda *_args: draft_input
            ),
            reqs=[],
            device=torch.device("cpu"),
            enable_overlap=False,
            input_ids=torch.empty((0,), dtype=torch.int64),
            spec_info=None,
        )
        server_args = SimpleNamespace(
            pp_size=1,
            speculative_num_draft_tokens=4,
        )

        ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(
            batch,
            server_args,
            future_map=None,
        )

        self.assertIs(batch.spec_info, draft_input)

    def test_dummy_tree_uses_bonus_tokens_as_roots(self):
        raw = EaglePPVerifyInputRaw.build_dummy_from_bonus_tokens(
            torch.tensor([7, 9], dtype=torch.int64), num_draft=4
        )

        self.assertTrue(
            torch.equal(raw.draft_tokens, torch.tensor([[7, 7, 7, 7], [9, 9, 9, 9]]))
        )
        self.assertTrue(
            torch.equal(raw.parent_list, torch.tensor([[-1, 0, 1], [-1, 0, 1]]))
        )
        self.assertTrue(
            torch.equal(raw.top_scores_index, torch.tensor([[0, 1, 2], [0, 1, 2]]))
        )
        self.assertTrue(torch.equal(raw.accept_lens, torch.tensor([1, 1])))
        self.assertIsNone(raw.accept_index)

    def test_raw_tree_filter_and_merge_preserve_tensor_order(self):
        raw = EaglePPVerifyInputRaw(
            draft_tokens=torch.tensor(
                [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
            ),
            bonus_tokens=torch.tensor([10, 20, 30]),
            top_scores_index=torch.tensor([[0, 1, 2]] * 3),
            parent_list=torch.tensor([[-1, 0, 1]] * 3),
            accept_lens=torch.tensor([1, 2, 3]),
            accept_index=torch.tensor([[0, 1], [2, 3], [4, 5]]),
        )
        raw.filter_batch(torch.tensor([2, 0]))

        self.assertTrue(torch.equal(raw.bonus_tokens, torch.tensor([30, 10])))
        self.assertTrue(torch.equal(raw.accept_lens, torch.tensor([3, 1])))
        self.assertTrue(torch.equal(raw.accept_index, torch.tensor([[4, 5], [0, 1]])))

        other = EaglePPVerifyInputRaw(
            draft_tokens=torch.tensor([[40, 41, 42, 43]]),
            bonus_tokens=torch.tensor([40]),
            top_scores_index=torch.tensor([[0, 1, 2]]),
            parent_list=torch.tensor([[-1, 0, 1]]),
            accept_lens=torch.tensor([4]),
            accept_index=torch.tensor([[6, 7]]),
        )
        raw.merge_batch(other)

        self.assertTrue(torch.equal(raw.bonus_tokens, torch.tensor([30, 10, 40])))
        self.assertTrue(torch.equal(raw.accept_lens, torch.tensor([3, 1, 4])))
        self.assertTrue(
            torch.equal(raw.accept_index, torch.tensor([[4, 5], [0, 1], [6, 7]]))
        )

    def test_worker_fallback_normalizes_direct_pd_handoff(self):
        bonus_tokens = torch.tensor([401, 402], dtype=torch.int64)
        batch = SimpleNamespace(
            spec_info=self._draft_input(bonus_tokens),
            input_ids=None,
        )
        worker = SimpleNamespace(speculative_num_draft_tokens=4)

        EAGLEWorkerV2._normalize_pp_verify_input_from_pd(worker, batch)

        self.assertTrue(torch.equal(batch.input_ids, bonus_tokens))
        self.assertIsInstance(batch.spec_info, EaglePPVerifyInputRaw)
        self.assertTrue(
            torch.equal(
                batch.spec_info.draft_tokens,
                torch.tensor([[401, 401, 401, 401], [402, 402, 402, 402]]),
            )
        )

    def test_pp_non_last_idle_does_not_require_draft_worker(self):
        worker = SimpleNamespace(
            _draft_worker=None,
            topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            device="cpu",
        )

        verify_input = EAGLEWorkerV2._build_idle_verify_input(worker, SimpleNamespace())

        self.assertTrue(verify_input.is_verify_input())
        self.assertEqual(verify_input.draft_token_num, 4)

    def test_non_pp_idle_draft_participates_and_builds_verify_input(self):
        draft = Mock()
        draft_worker = SimpleNamespace(
            draft=draft,
            draft_runner=SimpleNamespace(tp_group=object()),
            draft_tp_context=lambda _group: nullcontext(),
        )
        spec_algorithm = SimpleNamespace(is_standalone=lambda: False)
        worker = SimpleNamespace(
            _draft_worker=draft_worker,
            draft_worker=draft_worker,
            topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            device="cpu",
            speculative_algorithm=spec_algorithm,
            target_worker=SimpleNamespace(
                model_config=SimpleNamespace(vocab_size=1024)
            ),
            _pp_enabled=False,
        )
        batch = SimpleNamespace(
            global_num_tokens=torch.tensor([0, 1], dtype=torch.int64),
            spec_algorithm=spec_algorithm,
            spec_info=None,
            is_extend_in_batch=False,
        )

        with patch(
            "sglang.srt.speculative.eagle_worker_v2.get_draft_recurrent_hidden_state_spec",
            return_value=(4, torch.float32),
        ), patch(
            "sglang.srt.speculative.eagle_info.get_spec",
            return_value=SimpleNamespace(speculative_use_rejection_sampling=False),
        ), patch(
            "sglang.srt.speculative.eagle_worker_v2._should_force_symmetric_spec_moe_padding",
            return_value=True,
        ), patch(
            "sglang.srt.speculative.eagle_worker_v2.speculative_moe_backend_context",
            side_effect=nullcontext,
        ), patch(
            "sglang.srt.speculative.eagle_worker_v2.speculative_moe_a2a_backend_context",
            side_effect=nullcontext,
        ), patch(
            "sglang.srt.speculative.eagle_worker_v2.spec_stage_span",
            side_effect=lambda _name: nullcontext(),
        ):
            verify_input = EAGLEWorkerV2._build_idle_verify_input(worker, batch)

        draft.assert_called_once_with(batch)
        self.assertIsInstance(verify_input, EagleVerifyInput)
        self.assertTrue(verify_input.is_verify_input())
        self.assertEqual(verify_input.draft_token_num, 4)

    def test_pp_last_idle_skips_head_draft_and_builds_verify_input(self):
        draft = Mock()
        draft_worker = SimpleNamespace(draft=draft)
        worker = SimpleNamespace(
            _draft_worker=draft_worker,
            draft_worker=draft_worker,
            _pp_enabled=True,
            topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            device="cpu",
        )

        verify_input = EAGLEWorkerV2._build_idle_verify_input(worker, SimpleNamespace())

        draft.assert_not_called()
        self.assertIsInstance(verify_input, EagleVerifyInput)
        self.assertTrue(verify_input.is_verify_input())
        self.assertEqual(verify_input.draft_token_num, 4)

    def test_idle_without_symmetric_moe_skips_draft(self):
        draft = Mock()
        draft_worker = SimpleNamespace(
            draft=draft,
            draft_runner=SimpleNamespace(tp_group=object()),
            draft_tp_context=lambda _group: nullcontext(),
        )
        spec_algorithm = SimpleNamespace(is_standalone=lambda: False)
        worker = SimpleNamespace(
            _draft_worker=draft_worker,
            draft_worker=draft_worker,
            topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            device="cpu",
            speculative_algorithm=spec_algorithm,
            target_worker=SimpleNamespace(
                model_config=SimpleNamespace(vocab_size=1024)
            ),
            _pp_enabled=False,
        )
        batch = SimpleNamespace(
            global_num_tokens=torch.tensor([0, 0], dtype=torch.int64),
            spec_algorithm=spec_algorithm,
            spec_info=None,
            is_extend_in_batch=False,
        )

        with patch(
            "sglang.srt.speculative.eagle_worker_v2.get_draft_recurrent_hidden_state_spec",
            return_value=(4, torch.float32),
        ), patch(
            "sglang.srt.speculative.eagle_info.get_spec",
            return_value=SimpleNamespace(speculative_use_rejection_sampling=False),
        ), patch(
            "sglang.srt.speculative.eagle_worker_v2._should_force_symmetric_spec_moe_padding",
            return_value=False,
        ):
            verify_input = EAGLEWorkerV2._build_idle_verify_input(worker, batch)

        draft.assert_not_called()
        self.assertIsInstance(verify_input, EagleVerifyInput)
        self.assertTrue(verify_input.is_verify_input())

    def test_pp_raw_rebuild_uses_current_verify_mask_contract(self):
        raw = EaglePPVerifyInputRaw(
            draft_tokens=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]),
            bonus_tokens=torch.tensor([10, 20]),
            top_scores_index=torch.tensor([[0, 1, 2], [0, 1, 2]]),
            parent_list=torch.tensor([[-1, 0, 1], [-1, 0, 1]]),
            accept_lens=torch.tensor([2, 3]),
        )
        mask_buffer = torch.empty(32, dtype=torch.bool)
        verify_mask = SimpleNamespace(
            mode=TreeMaskMode.QLEN_ONLY,
            is_read=False,
            buffer=mask_buffer,
            fits=lambda bs: bs <= 8,
        )
        backend = SimpleNamespace(verify_mask=verify_mask, max_context_len=4096)
        worker = SimpleNamespace(
            topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            tree_mask_mode=TreeMaskMode.FULL_MASK,
            target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(attn_backend=backend)
            ),
        )
        batch = SimpleNamespace(
            spec_info=raw,
            seq_lens=torch.tensor([10, 12], dtype=torch.int64),
            seq_lens_cpu=None,
            seq_lens_sum=None,
            input_ids=None,
        )
        arranged = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23])
        kernel_result = (
            mask_buffer,
            torch.tensor([2]),
            torch.tensor([3]),
            torch.tensor([4]),
            torch.tensor([5]),
            arranged,
        )

        with patch(
            "sglang.srt.speculative.eagle_worker_v2.build_tree_kernel_efficient",
            return_value=kernel_result,
        ) as build_tree:
            verify = EAGLEWorkerV2._build_verify_input_from_pp_raw(worker, batch)

        self.assertEqual(build_tree.call_args.args[5], 0)
        self.assertIs(build_tree.call_args.args[10], mask_buffer)
        self.assertEqual(build_tree.call_args.args[9], TreeMaskMode.QLEN_ONLY)
        self.assertFalse(build_tree.call_args.kwargs["fill_prefix_mask"])
        self.assertEqual(
            build_tree.call_args.args[3].tolist(), [[11, 12, 13], [21, 22, 23]]
        )
        self.assertIs(batch.input_ids, arranged)
        self.assertEqual(verify.draft_token_num, 4)

    @patch(
        "sglang.srt.speculative.eagle_worker_v2.get_plan_stream",
        return_value=(object(), nullcontext()),
    )
    @patch("sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker")
    def test_pp_non_last_uses_target_war_runner(
        self, draft_worker_cls, _get_plan_stream
    ):
        server_args = SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_algorithm="EAGLE",
            speculative_adaptive=False,
            speculative_adaptive_config=None,
            pp_size=2,
            device="cpu",
            page_size=1,
            override=lambda *_args, **_kwargs: None,
        )
        target = SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=False),
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(context_len=4096),
                attn_backend=object(),
            ),
        )

        worker = EAGLEWorkerV2(
            server_args,
            gpu_id=0,
            ps=object(),
            nccl_port=1234,
            target_worker=target,
        )

        draft_worker_cls.assert_not_called()
        self.assertIsNone(worker.draft_worker)
        self.assertIs(worker.war_fastpath_runner, target.model_runner)


if __name__ == "__main__":
    unittest.main()
