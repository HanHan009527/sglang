import unittest
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    _restore_trtllm_decode_dp_padding,
    _trim_trtllm_decode_dp_padding,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSABackendDPPadding(unittest.TestCase):
    def test_trim_and_restore_real_metadata_rows(self):
        q = torch.arange(8 * 2).view(8, 2)
        topk = torch.arange(8 * 4, dtype=torch.int32).view(8, 4)

        real_q, real_topk, padding = _trim_trtllm_decode_dp_padding(
            q, topk, real_batch_size=6
        )

        self.assertTrue(torch.equal(real_q, q[:6]))
        self.assertTrue(torch.equal(real_topk, topk[:6]))
        self.assertEqual(padding, 2)

        output = _restore_trtllm_decode_dp_padding(torch.ones((6, 3)), padding)
        self.assertEqual(tuple(output.shape), (8, 3))
        self.assertTrue(torch.all(output[:6] == 1))
        self.assertTrue(torch.all(output[6:] == 0))

    def test_no_padding_preserves_inputs(self):
        q = torch.empty((2, 4))
        topk = torch.empty((2, 8), dtype=torch.int32)
        real_q, real_topk, padding = _trim_trtllm_decode_dp_padding(q, topk, 2)
        self.assertIs(real_q, q)
        self.assertIs(real_topk, topk)
        self.assertEqual(padding, 0)
        self.assertIs(_restore_trtllm_decode_dp_padding(q, 0), q)

    def test_metadata_rows_cannot_exceed_physical_rows(self):
        with self.assertRaisesRegex(AssertionError, "exceeds q batch size"):
            _trim_trtllm_decode_dp_padding(torch.empty((2, 4)), None, 3)

    def test_forward_trtllm_runs_real_rows_and_restores_padding(self):
        metadata = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([8, 12], dtype=torch.int32),
            page_table_1=torch.zeros((2, 12), dtype=torch.int32),
            max_seq_len_k=12,
        )
        backend = SimpleNamespace(
            forward_metadata=metadata,
            kv_cache_dtype=torch.bfloat16,
            token_to_kv_pool=SimpleNamespace(
                get_key_buffer=lambda _layer_id: torch.zeros((24, 3))
            ),
            real_page_size=1,
            kv_cache_dim=3,
            use_fused_topk=False,
            qk_nope_head_dim=2,
            kv_lora_rank=2,
            qk_rope_head_dim=1,
            workspace_buffer=None,
            dsa_index_topk=2,
            _multi_ctas_kv_counter_buffer=None,
            device="cpu",
            num_q_heads=2,
        )
        backend._pad_topk_indices = MethodType(
            DeepseekSparseAttnBackend._pad_topk_indices, backend
        )
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=2,
            head_dim=3,
            k_scale_float=None,
            scaling=1.0,
        )
        captured = {}
        flashinfer = ModuleType("flashinfer")
        decode = ModuleType("flashinfer.decode")

        def fake_decode(**kwargs):
            captured.update(kwargs)
            return torch.ones((2, 1, 2, 2), dtype=torch.bfloat16)

        decode.trtllm_batch_decode_with_kv_cache_mla = fake_decode
        flashinfer.decode = decode

        with (
            patch.dict(
                "sys.modules",
                {"flashinfer": flashinfer, "flashinfer.decode": decode},
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.transform_index_page_table_decode",
                return_value=torch.zeros((2, 2), dtype=torch.int32),
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.grow_multi_ctas_kv_counter_buffer_if_needed",
                return_value=None,
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.dsa_use_prefill_cp",
                return_value=False,
            ),
        ):
            output = DeepseekSparseAttnBackend._forward_trtllm(
                backend,
                q=torch.arange(4 * 2 * 3, dtype=torch.float32).view(4, 2, 3),
                k=torch.empty((4, 1, 2)),
                v=torch.empty((4, 1, 2)),
                layer=layer,
                forward_batch=SimpleNamespace(),
                seq_lens=metadata.cache_seqlens_int32,
                save_kv_cache=False,
                topk_indices=torch.arange(8, dtype=torch.int32).view(4, 2),
            )

        self.assertEqual(tuple(captured["query"].shape), (2, 1, 2, 3))
        self.assertTrue(torch.equal(captured["seq_lens"], metadata.cache_seqlens_int32))
        self.assertEqual(tuple(captured["block_tables"].shape), (2, 1, 2))
        self.assertEqual(tuple(output.shape), (4, 1, 2, 2))
        self.assertTrue(torch.all(output[:2] == 1))
        self.assertTrue(torch.all(output[2:] == 0))

    def test_forward_trtllm_fused_topk_uses_real_metadata_rows(self):
        metadata = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([5, 9], dtype=torch.int32),
            page_table_1=torch.zeros((2, 9), dtype=torch.int32),
            max_seq_len_k=9,
        )
        captured = {}
        backend = SimpleNamespace(
            forward_metadata=metadata,
            kv_cache_dtype=torch.bfloat16,
            token_to_kv_pool=SimpleNamespace(
                get_key_buffer=lambda _: torch.zeros((18, 3))
            ),
            real_page_size=1,
            kv_cache_dim=3,
            use_fused_topk=True,
            qk_nope_head_dim=2,
            kv_lora_rank=2,
            qk_rope_head_dim=1,
            workspace_buffer=None,
            dsa_index_topk=2,
            _multi_ctas_kv_counter_buffer=None,
            device="cpu",
            num_q_heads=2,
        )
        backend._pad_topk_indices = MethodType(
            DeepseekSparseAttnBackend._pad_topk_indices, backend
        )

        def fake_fused_topk(topk):
            captured["topk"] = topk
            return torch.zeros((2, 2), dtype=torch.int32)

        backend._get_fused_topk_page_table = fake_fused_topk
        layer = SimpleNamespace(
            layer_id=0, tp_q_head_num=2, head_dim=3, k_scale_float=None, scaling=1.0
        )
        flashinfer, decode = ModuleType("flashinfer"), ModuleType("flashinfer.decode")

        def fake_decode(**kwargs):
            captured.update(kwargs)
            return torch.full((2, 1, 2, 2), 3, dtype=torch.bfloat16)

        decode.trtllm_batch_decode_with_kv_cache_mla = fake_decode
        flashinfer.decode = decode
        with (
            patch.dict(
                "sys.modules", {"flashinfer": flashinfer, "flashinfer.decode": decode}
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.grow_multi_ctas_kv_counter_buffer_if_needed",
                return_value=None,
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.dsa_use_prefill_cp",
                return_value=False,
            ),
        ):
            output = DeepseekSparseAttnBackend._forward_trtllm(
                backend,
                torch.empty((4, 2, 3)),
                torch.empty((4, 1, 2)),
                torch.empty((4, 1, 2)),
                layer,
                SimpleNamespace(),
                metadata.cache_seqlens_int32,
                save_kv_cache=False,
                topk_indices=torch.tensor([[11, 12], [21, 22]], dtype=torch.int32),
            )
        self.assertTrue(
            torch.equal(
                captured["topk"], torch.tensor([[11, 12], [21, 22]], dtype=torch.int32)
            )
        )
        self.assertTrue(torch.equal(captured["seq_lens"], metadata.cache_seqlens_int32))
        self.assertEqual(tuple(output.shape), (4, 1, 2, 2))
        self.assertTrue(torch.all(output[:2] == 3))
        self.assertTrue(torch.all(output[2:] == 0))


if __name__ == "__main__":
    unittest.main()
