"""Source-isolated invariants for opt-in PP/DP phase tracing."""

import ast
import importlib.util
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock, patch

import torch

_ROOT = Path(__file__).parents[4]


class _CollectiveObserved(Exception):
    pass


_CI_REGISTER_PATH = _ROOT / "python/sglang/test/ci/ci_register.py"
_CI_SPEC = importlib.util.spec_from_file_location("ci_register", _CI_REGISTER_PATH)
ci_register = importlib.util.module_from_spec(_CI_SPEC)
assert _CI_SPEC.loader is not None
_CI_SPEC.loader.exec_module(ci_register)
register_cpu_ci = ci_register.register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _load_definition(
    path: Path, class_name: str | None = None, method_name: str | None = None
):
    """Compile the current source definition without importing SGLang runtime."""
    tree = ast.parse(path.read_text())
    node = next(
        item
        for item in tree.body
        if (
            isinstance(item, ast.ClassDef)
            if class_name is not None
            else isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        and item.name == (class_name if class_name is not None else method_name)
    )
    if class_name is not None and method_name is not None:
        node = next(
            item
            for item in node.body
            if isinstance(item, ast.FunctionDef) and item.name == method_name
        )
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        node.decorator_list = []
        node.returns = None
        for arg in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            arg.annotation = None
    return compile(
        ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
        str(path),
        "exec",
    )


def _load_mlp_sync_batch_info():
    namespace = {
        "dataclass": dataclass,
        "DPCooperationInfo": object,
        "ForwardMode": SimpleNamespace(IDLE=SimpleNamespace(value=0)),
        "Optional": Optional,
        "torch": torch,
    }
    path = _ROOT / "python/sglang/srt/managers/scheduler_components/dp_attn.py"
    exec(_load_definition(path, "MLPSyncBatchInfo"), namespace)
    return namespace["MLPSyncBatchInfo"], namespace


def test_dp_trace_does_not_change_collective_count_shape_or_order():
    sync_type, namespace = _load_mlp_sync_batch_info()
    group = object()
    active_ranks = torch.ones(1, dtype=torch.int64)
    namespace["_ENABLE_METRICS_DP_ATTENTION"] = False
    namespace["get_tp_group"] = lambda: SimpleNamespace(
        active_ranks_cpu=active_ranks, active_ranks=active_ranks
    )

    observations = []

    def all_gather(output, local, *, group):
        observations.append(
            ("all_gather", tuple(local.shape), tuple(output.shape), group)
        )
        raise _CollectiveObserved

    original = torch.distributed.all_gather_into_tensor
    torch.distributed.all_gather_into_tensor = Mock(side_effect=all_gather)
    try:
        for trace_enabled in (False, True):
            # The trace flag is intentionally external to MLPSyncBatchInfo. The
            # exact current class is executed for both modes to prove its wire
            # representation and collective call remain unchanged.
            namespace["phase_trace_enabled"] = trace_enabled
            sync_info = sync_type(
                dp_size=1,
                tp_size=1,
                cp_size=1,
                num_tokens=4,
                num_tokens_for_logprob=4,
                can_run_decode_cuda_graph=True,
                can_run_prefill_cuda_graph=True,
                can_run_draft_cuda_graph=False,
                is_extend_in_batch=False,
                local_can_run_tbo=True,
                local_forward_mode=1,
            )
            assert tuple(sync_info._get_local_tensor(device="cpu").shape) == (8,)
            before = len(observations)
            try:
                sync_info.all_gather(device="cpu", group=group)
            except _CollectiveObserved:
                pass
            else:
                raise AssertionError("all_gather_into_tensor was not called")
            assert observations[before:] == [("all_gather", (8,), (8,), group)]
    finally:
        torch.distributed.all_gather_into_tensor = original

    assert torch.distributed.all_gather_into_tensor is original
    assert observations == [
        ("all_gather", (8,), (8,), group),
        ("all_gather", (8,), (8,), group),
    ]


class _StopAfterCollective(Exception):
    pass


def test_prepare_mlp_sync_trace_preserves_actual_collective_call():
    path = _ROOT / "python/sglang/srt/managers/scheduler_components/dp_attn.py"
    code = _load_definition(path, method_name="prepare_mlp_sync_batch_raw")
    sync_type, _ = _load_mlp_sync_batch_info()
    results = []
    cpu_group = object()

    for trace_enabled in (False, True):
        events = []
        trace_emit = Mock(
            side_effect=lambda event, **_kwargs: events.append(f"trace:{event}")
        )
        batch_formatter = Mock(return_value={})
        pg_introspection = Mock(return_value={})

        class SyncInfo(sync_type):
            def all_gather(self, *, device, group, use_all_reduce):
                events.append(
                    (
                        "collective",
                        tuple(self._get_local_tensor(device=device).shape),
                        device,
                        group,
                        use_all_reduce,
                    )
                )
                raise _StopAfterCollective

        namespace = {
            "Callable": object,
            "MLPSyncBatchInfo": SyncInfo,
            "Optional": Optional,
            "ScheduleBatch": object,
            "SchedulerRecvSkipper": object,
            "TboDPAttentionPreparer": lambda: SimpleNamespace(
                prepare_all_gather=lambda _batch: (True, 1)
            ),
            "_log_mlp_sync_transport_once": Mock(),
            "_spec_diag_sync_logs": 0,
            "_spec_input_cuda_graph_compatible": Mock(return_value=True),
            "_use_device_mlp_sync_transport": Mock(return_value=False),
            "batch_phase_fields": batch_formatter,
            "check_cuda_graph_backend": Mock(return_value=False),
            "describe_process_group": pg_introspection,
            "envs": SimpleNamespace(
                SGLANG_SCHEDULER_SKIP_ALL_GATHER=SimpleNamespace(get=lambda: False)
            ),
            "get_parallel": Mock(),
            "logger": Mock(),
            "phase_tracer": SimpleNamespace(
                enabled=trace_enabled,
                allow_process_group_introspection=True,
                emit=trace_emit,
            ),
            "torch": torch,
            "world_dp_gather_enabled": Mock(return_value=False),
            "Phase": SimpleNamespace(PREFILL=object()),
            "Backend": SimpleNamespace(BREAKABLE=object()),
        }
        exec(code, namespace)
        tp_group = SimpleNamespace(
            cpu_group=cpu_group,
            device_group=object(),
            device="cpu",
            rank=0,
            rank_in_group=0,
            ranks=[0],
        )
        try:
            namespace["prepare_mlp_sync_batch_raw"](
                local_batch=None,
                model_runner=SimpleNamespace(prefill_cuda_graph_runner=None),
                dp_size=1,
                attn_tp_size=1,
                attn_cp_size=1,
                tp_group=tp_group,
                get_idle_batch=Mock(),
                disable_cuda_graph=False,
                require_mlp_tp_gather=False,
                disable_overlap_schedule=False,
                offload_tags=set(),
                phase_trace_context=(lambda: {"global_rank": 0}),
            )
        except _StopAfterCollective:
            pass
        else:
            raise AssertionError("actual prepare function did not call all_gather")

        results.append(
            (trace_enabled, events, trace_emit, batch_formatter, pg_introspection)
        )

    assert len(results[0][1]) == 1
    assert results[0][1][0][:3] == ("collective", (8,), "cpu")
    assert results[0][1][0][4] is False
    assert [
        event if isinstance(event, str) else event[0] for event in results[1][1]
    ] == [
        "trace:dp_all_gather_enter",
        "collective",
    ]
    assert results[1][1][1][1:] == results[0][1][0][1:]
    assert results[0][2].call_count == 0
    assert results[0][3].call_count == 0
    assert results[0][4].call_count == 0
    assert results[1][2].call_count == 1


class _ProxyTensors:
    def __init__(self, tensors):
        self.tensors = tensors


def _run_pp_exchange(trace_enabled: bool, pp_rank: int):
    path = _ROOT / "python/sglang/srt/managers/scheduler_pp_mixin.py"

    def emit(_event, *, collect=None, **_fields):
        if collect is not None:
            collect()

    phase_tracer = SimpleNamespace(
        enabled=trace_enabled,
        allow_process_group_introspection=True,
        emit=Mock(side_effect=emit),
    )
    describe_group = Mock(return_value={})
    namespace = {
        "PPProxyTensors": _ProxyTensors,
        "_pp_can_skip_output_comm": Mock(return_value=False),
        "batch_phase_fields": Mock(return_value={}),
        "describe_process_group": describe_group,
        "phase_tracer": phase_tracer,
        "torch": torch,
    }
    exec(
        _load_definition(
            path,
            "SchedulerPPMixin",
            "_pp_send_recv_and_preprocess_output_tensors",
        ),
        namespace,
    )
    exchange = namespace["_pp_send_recv_and_preprocess_output_tensors"]
    events = []
    target = SimpleNamespace(forward_mode=SimpleNamespace(is_prebuilt=lambda: False))
    scheduler = SimpleNamespace(
        ps=SimpleNamespace(pp_rank=pp_rank, attn_dp_rank=0, gpu_id=0),
        pp_group=SimpleNamespace(is_last_rank=pp_rank == 1),
        pp_output_group=SimpleNamespace(
            device_group=object(), ranks=[0, 1], rank=pp_rank, rank_in_group=pp_rank
        ),
        copy_stream_ctx=nullcontext(),
        copy_stream=SimpleNamespace(wait_stream=Mock()),
        schedule_stream=object(),
        device_module=SimpleNamespace(
            Event=Mock(return_value=Mock()), current_stream=Mock(return_value=object())
        ),
        _pp_send_output_to_next_stage=Mock(
            side_effect=lambda *_args: events.append("send") or []
        ),
        _pp_recv_dict_from_prev_stage=Mock(
            side_effect=lambda: events.append("recv") or {"next_token_ids": object()}
        ),
        _pp_prep_batch_result=Mock(return_value=object()),
    )

    exchange(
        scheduler,
        next_first_rank_mb_id=0,
        next_mb_id=0,
        mbs=[target],
        mb_metadata=[object()],
        last_rank_comm_queue=[],
        pp_outputs=None,
        relay_output_immediately=False,
    )
    return events, phase_tracer.emit, describe_group, namespace["batch_phase_fields"]


def _run_pp_immediate_exchange(trace_enabled: bool):
    path = _ROOT / "python/sglang/srt/managers/scheduler_pp_mixin.py"

    def emit(_event, *, collect=None, **_fields):
        if collect is not None:
            collect()

    phase_tracer = SimpleNamespace(
        enabled=trace_enabled,
        allow_process_group_introspection=True,
        emit=Mock(side_effect=emit),
    )
    namespace = {
        "PPProxyTensors": _ProxyTensors,
        "_pp_can_skip_output_comm": Mock(return_value=False),
        "batch_phase_fields": Mock(return_value={}),
        "describe_process_group": Mock(return_value={}),
        "phase_tracer": phase_tracer,
        "torch": torch,
    }
    exec(
        _load_definition(
            path,
            "SchedulerPPMixin",
            "_pp_send_recv_and_preprocess_output_tensors",
        ),
        namespace,
    )
    events = []
    send_work = [object()]
    received = {"next_token_ids": object()}
    scheduler = SimpleNamespace(
        ps=SimpleNamespace(pp_rank=0, attn_dp_rank=0, gpu_id=0),
        pp_group=SimpleNamespace(is_last_rank=False),
        pp_output_group=SimpleNamespace(
            device_group=object(), ranks=[0, 1], rank=0, rank_in_group=0
        ),
        copy_stream_ctx=nullcontext(),
        copy_stream=SimpleNamespace(
            wait_stream=lambda _stream: events.append("wait_stream")
        ),
        schedule_stream=SimpleNamespace(synchronize=lambda: events.append("fence")),
        device_module=SimpleNamespace(
            Event=lambda: SimpleNamespace(
                record=lambda _stream: events.append("record")
            ),
            current_stream=Mock(return_value=object()),
        ),
        _pp_recv_dict_from_prev_stage=Mock(
            side_effect=lambda: events.append("recv") or received
        ),
        _pp_prep_batch_result=Mock(
            side_effect=lambda *_args: events.append("prep") or object()
        ),
        _pp_send_dict_to_next_stage=Mock(
            side_effect=lambda *_args, **_kwargs: events.append("send") or send_work
        ),
        _pp_send_output_to_next_stage=Mock(),
        _pp_commit_comm_work=Mock(side_effect=lambda _work: events.append("commit")),
    )
    namespace["_pp_send_recv_and_preprocess_output_tensors"](
        scheduler,
        next_first_rank_mb_id=0,
        next_mb_id=0,
        mbs=[SimpleNamespace(forward_mode=SimpleNamespace(is_prebuilt=lambda: False))],
        mb_metadata=[object()],
        last_rank_comm_queue=[],
        pp_outputs=None,
        relay_output_immediately=True,
    )
    return events


def test_pp_trace_preserves_send_recv_order_and_default_off_is_zero_work():
    for pp_rank, expected in ((0, ["send", "recv"]), (1, ["recv", "send"])):
        off_events, off_log, off_pg, off_formatter = _run_pp_exchange(False, pp_rank)
        on_events, on_log, on_pg, on_formatter = _run_pp_exchange(True, pp_rank)

        assert off_events == on_events == expected
        assert off_log.call_count == off_pg.call_count == off_formatter.call_count == 0
        assert on_log.call_count > 0
        assert on_pg.call_count == on_log.call_count
        assert on_formatter.call_count > 0


def test_pp_immediate_trace_preserves_recv_fence_send_commit_order():
    expected = ["recv", "wait_stream", "prep", "record", "fence", "send", "commit"]
    assert _run_pp_immediate_exchange(False) == expected
    assert _run_pp_immediate_exchange(True) == expected


def _make_batch_result_processor(*, target_hisparse=None, draft_hisparse=None):
    from sglang.srt.managers.scheduler_components.batch_result_processor import (
        SchedulerBatchResultProcessor,
    )

    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(),
        model_config=SimpleNamespace(),
        token_to_kv_pool_allocator=Mock(),
        tree_cache=Mock(),
        hisparse_coordinator=target_hisparse,
        req_to_token_pool=Mock(),
        decode_offload_manager=None,
        metrics_collector=Mock(),
        metrics_reporter=Mock(),
        draft_worker=Mock(),
        model_worker=Mock(),
        logprob_result_processor=Mock(),
        output_streamer=Mock(),
        abort_request=Mock(),
        draft_hisparse_coordinator=draft_hisparse,
    )


def test_request_finish_trace_default_off_does_not_read_request_metadata():
    from sglang.srt.managers.scheduler_components import batch_result_processor

    class HostileReq:
        def __getattribute__(self, _name):
            raise AssertionError("disabled tracing read request metadata")

    processor = _make_batch_result_processor()
    with patch.object(batch_result_processor.phase_tracer, "enabled", False):
        processor._trace_request_finish("disabled", HostileReq())


def test_hisparse_finish_trace_preserves_target_then_draft_order():
    from sglang.srt.managers.scheduler_components import batch_result_processor

    calls = []
    target = SimpleNamespace(
        request_finished=lambda _req: calls.append("target_finish")
    )
    draft = SimpleNamespace(request_finished=lambda _req: calls.append("draft_finish"))
    processor = _make_batch_result_processor(
        target_hisparse=target, draft_hisparse=draft
    )
    req = SimpleNamespace(rid="r0", req_pool_idx=7, output_ids=[1, 2, 3])

    def emit(event, **_fields):
        calls.append(event)

    with (
        patch.object(batch_result_processor.phase_tracer, "enabled", True),
        patch.object(batch_result_processor.phase_tracer, "emit", side_effect=emit),
    ):
        processor._finish_hisparse_request(req)

    assert calls == [
        "request_finish_hisparse_target_before",
        "target_finish",
        "request_finish_hisparse_target_after",
        "request_finish_hisparse_draft_before",
        "draft_finish",
        "request_finish_hisparse_draft_after",
    ]
