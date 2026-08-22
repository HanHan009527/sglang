"""Regression coverage for EAGLE idle ranks with symmetric MoE A2A."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.dsa_indexer import (
    Indexer,
    _is_logical_eager_idle,
)
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.managers.scheduler_components.dp_attn import _update_gather_batch
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    _should_bypass_attention_for_symmetric_spec_moe_dummy,
    _should_force_symmetric_spec_moe_padding,
    _should_materialize_idle_spec_moe,
    requires_symmetric_spec_deepep_lockstep,
)
from sglang.srt.utils.common import require_mlp_tp_gather
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _sync_info(global_num_tokens):
    return SimpleNamespace(
        num_tokens=0,
        num_tokens_for_logprob=0,
        global_num_tokens=global_num_tokens,
        global_num_tokens_for_logprob=global_num_tokens,
        is_extend_in_batch=False,
        tbo_split_seq_index=None,
        global_forward_mode=None,
        can_run_decode_cuda_graph=False,
        can_run_prefill_cuda_graph=False,
        can_run_draft_cuda_graph=True,
    )


def _symmetric_backend_patches(backend_name):
    backend = SimpleNamespace(
        is_deepep=lambda: backend_name == "deepep",
        is_megamoe=lambda: backend_name == "megamoe",
        is_pplx=lambda: False,
    )
    mode = SimpleNamespace(
        resolve=lambda _is_extend: SimpleNamespace(is_low_latency=lambda: True)
    )
    return (
        patch(
            "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
            return_value=backend,
        ),
        patch("sglang.srt.layers.moe.utils.get_deepep_mode", return_value=mode),
    )


class TestSymmetricMoeSpecIdlePadding(CustomTestCase):
    def test_only_idle_draft_rows_bypass_attention_for_symmetric_moe(self):
        draft_info = SimpleNamespace(is_draft_input=lambda: True)
        verify_info = SimpleNamespace(is_draft_input=lambda: False)

        self.assertTrue(
            _should_bypass_attention_for_symmetric_spec_moe_dummy(
                forward_mode=ForwardMode.IDLE,
                spec_info=draft_info,
                force_symmetric_spec_moe_padding=True,
            )
        )
        for mode, spec_info, force_symmetric in (
            (ForwardMode.DRAFT_EXTEND_V2, draft_info, True),
            (ForwardMode.IDLE, verify_info, True),
            (ForwardMode.IDLE, draft_info, False),
        ):
            self.assertFalse(
                _should_bypass_attention_for_symmetric_spec_moe_dummy(
                    forward_mode=mode,
                    spec_info=spec_info,
                    force_symmetric_spec_moe_padding=force_symmetric,
                )
            )

    def test_eagle_mixed_active_idle_keeps_peer_counts(self):
        batch = SimpleNamespace(spec_algorithm=SimpleNamespace(is_eagle=lambda: True))
        sync_info = _sync_info([6, 0, 6, 6])

        for backend_name in ("deepep", "megamoe"):
            backend_patch, mode_patch = _symmetric_backend_patches(backend_name)
            with self.subTest(backend=backend_name), backend_patch, mode_patch:
                _update_gather_batch(
                    batch,
                    sync_info,
                    require_mlp_tp_gather=False,
                )

                self.assertEqual(batch.global_num_tokens, [6, 0, 6, 6])
                self.assertEqual(batch.global_num_tokens_for_logprob, [6, 0, 6, 6])

    def test_non_spec_symmetric_backend_keeps_rank_local_counts(self):
        batch = SimpleNamespace(spec_algorithm=None)
        sync_info = _sync_info([6, 0, 6, 6])

        for backend_name in ("deepep", "megamoe"):
            backend_patch, mode_patch = _symmetric_backend_patches(backend_name)
            with self.subTest(backend=backend_name), backend_patch, mode_patch:
                _update_gather_batch(
                    batch,
                    sync_info,
                    require_mlp_tp_gather=False,
                )

                self.assertEqual(batch.global_num_tokens, [0])
                self.assertEqual(batch.global_num_tokens_for_logprob, [0])

    def test_idle_spec_materializes_only_under_symmetric_padding(self):
        verify_info = SimpleNamespace(is_draft_input=lambda: False)
        draft_info = SimpleNamespace(is_draft_input=lambda: True)

        self.assertTrue(
            _should_materialize_idle_spec_moe(
                forward_mode=ForwardMode.IDLE,
                spec_info=verify_info,
                dp_padding_mode=DpPaddingMode.MAX_LEN,
                num_tokens=6,
            )
        )
        for spec_info, padding, num_tokens in (
            (verify_info, DpPaddingMode.SUM_LEN, 6),
            (verify_info, DpPaddingMode.MAX_LEN, 0),
        ):
            self.assertFalse(
                _should_materialize_idle_spec_moe(
                    forward_mode=ForwardMode.IDLE,
                    spec_info=spec_info,
                    dp_padding_mode=padding,
                    num_tokens=num_tokens,
                )
            )
        self.assertTrue(
            _should_materialize_idle_spec_moe(
                forward_mode=ForwardMode.IDLE,
                spec_info=draft_info,
                dp_padding_mode=DpPaddingMode.MAX_LEN,
                num_tokens=1,
            )
        )

    def test_sparse_eagle_verify_forces_symmetric_moe_padding(self):
        algorithm = SimpleNamespace(is_eagle=lambda: True)
        verify_info = SimpleNamespace(is_draft_input=lambda: False)
        draft_info = SimpleNamespace(is_draft_input=lambda: True)
        for backend_name in ("deepep", "megamoe"):
            backend_patch, mode_patch = _symmetric_backend_patches(backend_name)
            with self.subTest(backend=backend_name), backend_patch, mode_patch:
                self.assertTrue(
                    _should_force_symmetric_spec_moe_padding(
                        spec_algorithm=algorithm,
                        spec_info=verify_info,
                        is_extend_in_batch=False,
                        global_num_tokens=[0, 6, 6, 0, 0, 0, 0, 0],
                    )
                )
                self.assertTrue(
                    _should_force_symmetric_spec_moe_padding(
                        spec_algorithm=algorithm,
                        spec_info=draft_info,
                        is_extend_in_batch=False,
                        global_num_tokens=[0, 1, 1, 0, 0, 0, 0, 0],
                    )
                )

    def test_symmetric_padding_is_limited_to_mixed_eagle_rounds(self):
        eagle = SimpleNamespace(is_eagle=lambda: True)
        non_eagle = SimpleNamespace(is_eagle=lambda: False)
        spec_info = SimpleNamespace(is_draft_input=lambda: False)
        backend_patch, mode_patch = _symmetric_backend_patches("megamoe")

        with backend_patch, mode_patch:
            for algorithm, info, counts in (
                (eagle, spec_info, [1, 1, 1, 1]),
                (eagle, spec_info, [0, 0, 0, 0]),
                (non_eagle, spec_info, [0, 1, 0, 0]),
                (eagle, None, [0, 1, 0, 0]),
            ):
                with self.subTest(
                    is_eagle=algorithm.is_eagle(),
                    has_spec_info=info is not None,
                    counts=counts,
                ):
                    self.assertFalse(
                        _should_force_symmetric_spec_moe_padding(
                            spec_algorithm=algorithm,
                            spec_info=info,
                            is_extend_in_batch=False,
                            global_num_tokens=counts,
                        )
                    )

    def test_megamoe_does_not_enable_mlp_tp_gather(self):
        parallel = SimpleNamespace(
            enable_dp_attention=True,
            dp_size=8,
            moe_dense_tp_size=1,
            enable_dp_lm_head=True,
        )
        execution = SimpleNamespace(moe=SimpleNamespace(elastic_ep_backend=None))
        backend = SimpleNamespace(
            is_none=lambda: False,
            is_flashinfer=lambda: False,
            is_megamoe=lambda: True,
        )
        server_args = SimpleNamespace(tp_size=8)

        with (
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
            patch("sglang.srt.runtime_context.get_exec", return_value=execution),
            patch(
                "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
                return_value=backend,
            ),
        ):
            self.assertFalse(require_mlp_tp_gather(server_args))

    def test_idle_megamoe_spec_materialization_round_trip(self):
        spec_info = SimpleNamespace(
            is_draft_input=lambda: True,
            num_tokens_per_req=1,
            hidden_states=torch.empty(0, 8),
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.IDLE,
            batch_size=0,
            input_ids=torch.empty(0, dtype=torch.int64),
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            seq_lens=torch.empty(0, dtype=torch.int64),
            out_cache_loc=torch.empty(0, dtype=torch.int64),
            seq_lens_sum=0,
            positions=torch.empty(0, dtype=torch.int64),
            seq_lens_cpu=torch.empty(0, dtype=torch.int64),
            lora_ids=[],
            spec_info=spec_info,
            spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
            is_extend_in_batch=False,
            original_global_num_tokens_cpu=[0, 1, 1, 0],
            global_num_tokens_cpu=[0, 1, 1, 0],
            global_num_tokens_for_logprob_cpu=[0, 1, 1, 0],
            global_num_tokens_gpu=torch.zeros(4, dtype=torch.int64),
            num_token_non_padded=torch.tensor(0, dtype=torch.int32),
            num_token_non_padded_cpu=0,
        )
        model_runner = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(mtp_hybrid_override_pattern=None)
            ),
            is_draft_worker=True,
            server_args=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    prefill=SimpleNamespace(bs=[]),
                )
            ),
            attn_backend=SimpleNamespace(
                get_cuda_graph_seq_len_fill_value=lambda: 1,
            ),
        )
        parallel = SimpleNamespace(attn_tp_size=1, attn_dp_rank=0)
        backend_patch, mode_patch = _symmetric_backend_patches("megamoe")

        with (
            backend_patch,
            mode_patch,
            patch(
                "sglang.srt.model_executor.forward_batch_info.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.model_executor.forward_batch_info.mambaish_config",
                return_value=None,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_attention_dp_size",
                return_value=4,
            ),
            patch("sglang.srt.layers.cp.utils.enable_cp_v2", return_value=True),
            patch("sglang.srt.model_executor.forward_batch_info.set_dp_buffer_len"),
            patch(
                "sglang.srt.model_executor.forward_batch_info.set_is_extend_in_batch"
            ),
            patch(
                "sglang.srt.batch_overlap.two_batch_overlap.TboForwardBatchPreparer.prepare"
            ),
        ):
            forward_batch.prepare_mlp_sync_batch(model_runner)

        self.assertEqual(forward_batch._original_forward_mode, ForwardMode.IDLE)
        self.assertEqual(forward_batch.forward_mode, ForwardMode.DECODE)
        self.assertEqual(forward_batch.batch_size, 1)
        self.assertEqual(forward_batch.input_ids.numel(), 1)
        self.assertEqual(forward_batch.num_token_non_padded_cpu, 1)
        self.assertEqual(forward_batch.num_token_non_padded.item(), 1)
        self.assertEqual(forward_batch.global_num_tokens_cpu, [1, 1, 1, 1])
        self.assertTrue(forward_batch.dp_padding_mode.is_max_len())
        self.assertTrue(forward_batch.symmetric_spec_moe_dummy)

        logits_output = SimpleNamespace(
            next_token_logits=torch.zeros(1, 8),
            hidden_states=torch.zeros(1, 8),
        )
        forward_batch.post_forward_mlp_sync_batch(logits_output)
        self.assertEqual(forward_batch.forward_mode, ForwardMode.IDLE)
        self.assertEqual(forward_batch.batch_size, 0)
        self.assertEqual(forward_batch.positions.numel(), 0)
        self.assertEqual(logits_output.next_token_logits.shape[0], 0)
        self.assertEqual(logits_output.hidden_states.shape[0], 0)

    def test_idle_megamoe_verify_preserves_logical_idle_marker(self):
        spec_info = SimpleNamespace(
            is_draft_input=lambda: False,
            num_tokens_per_req=4,
            draft_token_num=4,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.IDLE,
            batch_size=0,
            input_ids=torch.empty(0, dtype=torch.int64),
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            seq_lens=torch.empty(0, dtype=torch.int64),
            out_cache_loc=torch.empty(0, dtype=torch.int64),
            seq_lens_sum=0,
            positions=torch.empty(0, dtype=torch.int64),
            seq_lens_cpu=torch.empty(0, dtype=torch.int64),
            lora_ids=[],
            spec_info=spec_info,
            spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
            is_extend_in_batch=False,
            original_global_num_tokens_cpu=[1, 1, 1, 1, 0, 1, 1, 0],
            global_num_tokens_cpu=[4, 4, 4, 4, 0, 4, 4, 0],
            global_num_tokens_for_logprob_cpu=[4, 4, 4, 4, 0, 4, 4, 0],
            global_num_tokens_gpu=torch.zeros(8, dtype=torch.int64),
            num_token_non_padded=torch.tensor(0, dtype=torch.int32),
            num_token_non_padded_cpu=0,
        )
        model_runner = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(mtp_hybrid_override_pattern=None)
            ),
            is_draft_worker=False,
            server_args=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    prefill=SimpleNamespace(bs=[]),
                )
            ),
            attn_backend=SimpleNamespace(
                get_cuda_graph_seq_len_fill_value=lambda: 1,
            ),
        )
        parallel = SimpleNamespace(attn_tp_size=1, attn_dp_rank=4)
        backend_patch, mode_patch = _symmetric_backend_patches("megamoe")

        with (
            backend_patch,
            mode_patch,
            patch(
                "sglang.srt.model_executor.forward_batch_info.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.model_executor.forward_batch_info.mambaish_config",
                return_value=None,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_attention_dp_size",
                return_value=8,
            ),
            patch("sglang.srt.layers.cp.utils.enable_cp_v2", return_value=True),
            patch("sglang.srt.model_executor.forward_batch_info.set_dp_buffer_len"),
            patch(
                "sglang.srt.model_executor.forward_batch_info.set_is_extend_in_batch"
            ),
            patch(
                "sglang.srt.batch_overlap.two_batch_overlap.TboForwardBatchPreparer.prepare"
            ),
        ):
            forward_batch.prepare_mlp_sync_batch(model_runner)

        self.assertEqual(forward_batch._original_forward_mode, ForwardMode.IDLE)
        self.assertEqual(forward_batch.forward_mode, ForwardMode.TARGET_VERIFY)
        self.assertEqual(forward_batch.batch_size, 1)
        self.assertEqual(forward_batch.input_ids.numel(), 4)
        self.assertEqual(forward_batch.num_token_non_padded_cpu, 4)
        self.assertEqual(forward_batch.num_token_non_padded.item(), 4)
        self.assertEqual(forward_batch.global_num_tokens_cpu, [4] * 8)
        self.assertTrue(forward_batch.dp_padding_mode.is_max_len())
        # This flag is intentionally draft-only: verify rows still run target
        # model MLP/MoE and only the DSA indexer uses _original_forward_mode
        # to recognize that they do not own request/KV rows.
        self.assertFalse(forward_batch.symmetric_spec_moe_dummy)
        self.assertTrue(_is_logical_eager_idle(forward_batch))

        indexer = SimpleNamespace(index_topk=8)
        with (
            patch("sglang.srt.layers.attention.dsa.dsa_indexer._is_cuda", True),
            patch(
                "sglang.srt.layers.attention.dsa.dsa_indexer.get_is_capture_mode",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.attention.dsa.dsa_indexer._broadcast_indexer_topk_from_rank0",
                side_effect=lambda result: result,
            ),
            patch(
                "sglang.srt.layers.attention.dsa.dsa_indexer.maybe_capture_indexer_topk",
                side_effect=lambda _layer_id, result: result,
            ),
            patch(
                "sglang.srt.layers.attention.dsa.dsa_indexer.get_attn_backend",
                side_effect=AssertionError(
                    "logical idle verify must not fetch DSA metadata"
                ),
            ),
        ):
            topk_result = Indexer.forward_cuda(
                indexer,
                x=torch.empty((4, 16)),
                q_lora=torch.empty((4, 16)),
                positions=forward_batch.positions,
                forward_batch=forward_batch,
                layer_id=0,
            )

        self.assertEqual(topk_result.shape, (4, 8))
        self.assertTrue(torch.all(topk_result == -1))

    def test_only_mixed_active_idle_spec_requires_lockstep(self):
        shared = dict(
            spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
            spec_info=SimpleNamespace(is_draft_input=lambda: False),
            is_extend_in_batch=False,
            dp_padding_mode=DpPaddingMode.MAX_LEN,
            original_global_num_tokens_cpu=[0, 1, 0, 0],
        )
        active_draft_extend_batch = SimpleNamespace(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2, **shared
        )
        dummy_decode_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            **{**shared, "spec_info": SimpleNamespace(is_draft_input=lambda: True)},
        )
        target_verify_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY, **shared
        )

        backend_patch, mode_patch = _symmetric_backend_patches("deepep")
        with backend_patch, mode_patch:
            self.assertTrue(
                requires_symmetric_spec_deepep_lockstep(active_draft_extend_batch)
            )
            self.assertTrue(requires_symmetric_spec_deepep_lockstep(dummy_decode_batch))
            self.assertTrue(
                requires_symmetric_spec_deepep_lockstep(target_verify_batch)
            )

            for unsupported_mode in (
                ForwardMode.MIXED,
                ForwardMode.IDLE,
                ForwardMode.EXTEND,
                ForwardMode.PREBUILT,
                ForwardMode.SPLIT_PREFILL,
                ForwardMode.DLLM_EXTEND,
            ):
                unsupported_batch = SimpleNamespace(
                    forward_mode=unsupported_mode, **shared
                )
                self.assertFalse(
                    requires_symmetric_spec_deepep_lockstep(unsupported_batch)
                )

            active_draft_extend_batch.original_global_num_tokens_cpu = [1, 1, 1, 1]
            self.assertFalse(
                requires_symmetric_spec_deepep_lockstep(active_draft_extend_batch)
            )
            active_draft_extend_batch.original_global_num_tokens_cpu = [0, 0, 0, 0]
            self.assertFalse(
                requires_symmetric_spec_deepep_lockstep(active_draft_extend_batch)
            )

        active_draft_extend_batch.original_global_num_tokens_cpu = [0, 1, 0, 0]
        backend_patch, mode_patch = _symmetric_backend_patches("megamoe")
        with backend_patch, mode_patch:
            self.assertFalse(
                requires_symmetric_spec_deepep_lockstep(active_draft_extend_batch)
            )


if __name__ == "__main__":
    unittest.main()
