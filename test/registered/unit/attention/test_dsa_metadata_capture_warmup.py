from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

import sglang.kernels.ops.attention.dsa_metadata as dsa_metadata_module
import sglang.srt.layers.attention.dsa_backend as dsa_backend_module
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_first_graph_bucket_launches_fused_target_verify_metadata(monkeypatch):
    """The capture-time build must also load the fused replay kernel.

    A new graph bucket used to return immediately after allocating metadata. The
    first real replay then paid Triton's lazy ``cuModuleLoadData`` cost in the
    distributed request path. This CPU test mocks the CUDA launch but pins the
    first-call dispatch contract.
    """
    bs = 2
    next_n = 4
    max_len = 8
    expanded_bs = bs * next_n

    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend.decode_cuda_graph_metadata = {}
    backend.req_to_token = torch.arange(4 * max_len, dtype=torch.int32).view(4, max_len)
    backend.device = torch.device("cpu")
    backend.dsa_index_topk = 4
    backend.real_page_size = 1
    backend.speculative_num_draft_tokens = next_n
    backend.dsa_decode_impl = "triton"
    backend.set_dsa_prefill_impl = MagicMock()
    backend._refresh_paged_mqa_schedule_metadata = MagicMock()
    backend._refresh_topk_v2_plan = MagicMock()
    backend._build_paged_mqa_schedule_2d_ctx_lens = MagicMock(
        side_effect=lambda _mode, cache_lens, _expanded, _bs: cache_lens.view(-1, 1)
    )

    page_table = torch.empty((expanded_bs, max_len), dtype=torch.int32)
    metadata = SimpleNamespace(
        cache_seqlens_int32=torch.empty(bs, dtype=torch.int32),
        cu_seqlens_k=torch.empty(bs + 1, dtype=torch.int32),
        page_table_1=page_table,
        dsa_seqlens_expanded=torch.empty(expanded_bs, dtype=torch.int32),
        dsa_cache_seqlens_int32=torch.empty(expanded_bs, dtype=torch.int32),
        dsa_cu_seqlens_k=torch.empty(expanded_bs + 1, dtype=torch.int32),
        real_page_table=page_table,
        paged_mqa_ctx_lens_2d=None,
        page_size=1,
    )
    build_calls = []

    def fake_build(bucket_bs, *_args, **_kwargs):
        build_calls.append(bucket_bs)
        backend.decode_cuda_graph_metadata[bucket_bs] = metadata
        backend.forward_metadata = metadata

    backend._build_forward_metadata_cuda_graph = fake_build

    launch_calls = []

    def fake_fused_target_verify_metadata(**kwargs):
        launch_calls.append(kwargs["bs"])
        cache_lens = kwargs["seq_lens"].to(torch.int32) + kwargs["next_n"]
        kwargs["cache_seqlens"].copy_(cache_lens)
        kwargs["seqlens_expanded"].copy_(cache_lens.repeat_interleave(kwargs["next_n"]))
        kwargs["dsa_cache_seqlens"].copy_(
            kwargs["seqlens_expanded"].clamp_max(kwargs["dsa_index_topk"])
        )

    monkeypatch.setattr(dsa_backend_module, "is_cuda", lambda: True)
    monkeypatch.setattr(dsa_backend_module, "_is_hip", False)
    monkeypatch.setattr(dsa_backend_module, "is_sm100_supported", lambda: False)
    monkeypatch.setattr(
        dsa_metadata_module,
        "fused_dsa_target_verify_metadata",
        fake_fused_target_verify_metadata,
    )

    seq_lens = torch.tensor([3, 5], dtype=torch.int64)
    backend._apply_cuda_graph_metadata(
        bs=bs,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=None,
    )

    assert build_calls == [bs]
    assert launch_calls == [bs]
    assert backend.forward_metadata is metadata
