import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.scheduler_components.dp_attn import MLPSyncBatchInfo
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import EaglePPVerifyInputRaw
from sglang.srt.speculative.eagle_worker_v2 import (
    EAGLEWorkerV2,
    _materialize_symmetric_idle_dsa_seed,
    _pp_tail_draft_forward_mode,
    _require_pp_tail_dsa_seed_for_graph,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class TestEaglePDDPFallback(CustomTestCase):
    def test_draft_graph_gate_has_independent_dp_vote(self):
        sync_info = MLPSyncBatchInfo(
            dp_size=1,
            tp_size=1,
            cp_size=1,
            num_tokens=1,
            num_tokens_for_logprob=1,
            can_run_decode_cuda_graph=True,
            can_run_prefill_cuda_graph=False,
            can_run_draft_cuda_graph=False,
            is_extend_in_batch=False,
            local_can_run_tbo=True,
            local_forward_mode=ForwardMode.DECODE.value,
        )

        local = sync_info._get_local_tensor(device="cpu")
        fallback = sync_info._get_fallback_tensor(device="cpu")
        self.assertEqual(local[2].item(), 1)
        self.assertEqual(local[7].item(), 0)
        # Idle/inactive DP ranks must not veto an active rank's draft vote.
        self.assertEqual(fallback[7].item(), 1)

    def test_seedless_gate_only_disables_draft_graph(self):
        runner = object.__new__(EAGLEDraftCudaGraphRunner)
        runner.require_mlp_tp_gather = False
        runner.require_mlp_sync = True
        runner.disable_padding = False
        runner.captured_req_width = 1
        runner.max_bs = 8

        forward_batch = SimpleNamespace(
            spec_info=SimpleNamespace(num_tokens_per_req=1),
            batch_size=1,
            can_run_dp_cuda_graph=True,
            can_run_dp_draft_cuda_graph=False,
        )
        self.assertFalse(runner.can_run_graph(forward_batch))

        # Target verify and draft-extend only consume the ordinary gate.
        forward_batch.can_run_dp_draft_cuda_graph = True
        self.assertTrue(runner.can_run_graph(forward_batch))

    def test_seedless_pd_draft_requests_rank_consistent_eager_forward(self):
        worker = object.__new__(EAGLEWorkerV2)
        worker._draft_worker = SimpleNamespace(seed_dsa_topk_from_draft_extend=True)
        worker._pp_enabled = False

        for seed, future_indices, future_seed, expect_eager in (
            (None, None, False, True),
            (torch.ones((1, 1)), None, False, False),
            (torch.ones((1, 1)), torch.tensor([1]), False, True),
            (None, torch.tensor([1]), True, False),
        ):
            with self.subTest(
                seed_present=seed is not None,
                overlap=future_indices is not None,
                future_seed=future_seed,
            ):
                batch = SimpleNamespace(
                    spec_info=SimpleNamespace(
                        dsa_topk_indices=seed,
                        future_indices=future_indices,
                        future_dsa_topk_indices_available=future_seed,
                    )
                )
                self.assertEqual(
                    worker.requires_dp_attention_eager_forward(batch),
                    expect_eager,
                )

        worker._draft_worker.seed_dsa_topk_from_draft_extend = False
        self.assertFalse(
            worker.requires_dp_attention_eager_forward(
                SimpleNamespace(spec_info=SimpleNamespace(dsa_topk_indices=None))
            )
        )

    def test_non_last_pp_rank_is_permissive(self):
        worker = object.__new__(EAGLEWorkerV2)
        worker._draft_worker = None
        self.assertFalse(
            worker.requires_dp_attention_eager_forward(
                SimpleNamespace(spec_info=SimpleNamespace(dsa_topk_indices=None))
            )
        )

    def test_pp_last_votes_for_post_extend_tail_draft(self):
        worker = object.__new__(EAGLEWorkerV2)
        worker._draft_worker = SimpleNamespace(seed_dsa_topk_from_draft_extend=True)
        worker._pp_enabled = True
        worker._pp_is_last_rank = True

        raw = EaglePPVerifyInputRaw.build_dummy_from_bonus_tokens(
            torch.tensor([7], dtype=torch.int64), num_draft=4
        )
        self.assertFalse(
            worker.requires_dp_attention_eager_forward(SimpleNamespace(spec_info=raw))
        )

        # The actual PP tail draft consumes the input produced after verify by
        # draft-extend, not the raw verify tree inspected above.
        post_extend = SimpleNamespace(
            dsa_topk_indices=torch.ones((1, 2), dtype=torch.int32)
        )
        self.assertFalse(
            worker.requires_dp_attention_eager_forward(
                SimpleNamespace(spec_info=post_extend)
            )
        )

    def test_symmetric_idle_dsa_seed_matches_dummy_graph_batch(self):
        draft_input = SimpleNamespace(
            dsa_topk_indices=torch.empty((0, 2), dtype=torch.int32)
        )

        _materialize_symmetric_idle_dsa_seed(draft_input)

        self.assertEqual(draft_input.dsa_topk_indices.shape, (1, 2))
        self.assertEqual(draft_input.dsa_topk_indices.dtype, torch.int32)
        self.assertTrue(
            torch.equal(
                draft_input.dsa_topk_indices, torch.zeros((1, 2), dtype=torch.int32)
            )
        )

        runner = object.__new__(EAGLEDraftCudaGraphRunner)
        runner.buffers = SimpleNamespace(
            dsa_seed_topk=torch.empty((64, 2), dtype=torch.int32)
        )
        runner._validate_dsa_seed_topk(SimpleNamespace(spec_info=draft_input), raw_bs=1)

    def test_pp_active_tail_requires_post_extend_seed_for_graph(self):
        for graph, mode, seed, expect_error in (
            (True, ForwardMode.DECODE, None, True),
            (True, ForwardMode.DECODE, torch.ones((1, 2)), False),
            (False, ForwardMode.DECODE, None, False),
            (True, ForwardMode.IDLE, None, False),
        ):
            with self.subTest(
                graph=graph,
                mode=mode.name,
                seed_present=seed is not None,
            ):
                kwargs = dict(
                    pp_size=2,
                    can_run_decode_cuda_graph=graph,
                    forward_mode=mode,
                    seed_dsa_topk_from_draft_extend=True,
                    draft_input=SimpleNamespace(dsa_topk_indices=seed),
                )
                if expect_error:
                    with self.assertRaisesRegex(
                        RuntimeError, "PP tail draft selected CUDA Graph"
                    ):
                        _require_pp_tail_dsa_seed_for_graph(**kwargs)
                else:
                    _require_pp_tail_dsa_seed_for_graph(**kwargs)

    def test_pp_tail_draft_preserves_idle_mode(self):
        self.assertEqual(
            _pp_tail_draft_forward_mode(ForwardMode.IDLE), ForwardMode.IDLE
        )
        self.assertEqual(
            _pp_tail_draft_forward_mode(ForwardMode.DECODE), ForwardMode.DECODE
        )
        self.assertEqual(
            _pp_tail_draft_forward_mode(ForwardMode.TARGET_VERIFY),
            ForwardMode.DECODE,
        )


if __name__ == "__main__":
    unittest.main()
